"""SQLite 标注库及版本化审核记录。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from anitopy_ml.errors import ConfigurationError, SchemaValidationError
from anitopy_ml.schemas import AnnotationRecord


class AnnotationStore:
    """保存样本、标注版本、扁平片段、结构字段、来源和审计信息。"""

    def __init__(self, database: str | Path) -> None:
        """打开数据库；文件数据库的父目录不存在时自动创建。"""
        if str(database) != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        """关闭数据库连接。"""
        self.connection.close()

    def __enter__(self) -> AnnotationStore:
        """支持上下文管理器。"""
        return self

    def __exit__(self, *_: object) -> None:
        """离开上下文时关闭数据库。"""
        self.close()

    def _create_schema(self) -> None:
        """初始化第一版所需的表和索引。"""
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                raw_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS annotation_versions (
                sample_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                tier TEXT NOT NULL,
                review_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sample_id, version),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            );
            CREATE TABLE IF NOT EXISTS spans (
                sample_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                position INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                label TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY (sample_id, version, position),
                FOREIGN KEY (sample_id, version) REFERENCES annotation_versions(sample_id, version)
            );
            CREATE TABLE IF NOT EXISTS field_values (
                sample_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                extracted_json TEXT NOT NULL,
                catalog_json TEXT,
                PRIMARY KEY (sample_id, version),
                FOREIGN KEY (sample_id, version) REFERENCES annotation_versions(sample_id, version)
            );
            CREATE TABLE IF NOT EXISTS lineage (
                sample_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                sources_json TEXT NOT NULL,
                training_allowed INTEGER NOT NULL,
                authorization_ref TEXT,
                PRIMARY KEY (sample_id, version),
                FOREIGN KEY (sample_id, version) REFERENCES annotation_versions(sample_id, version)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id, version) REFERENCES annotation_versions(sample_id, version)
            );
            CREATE TABLE IF NOT EXISTS work_group_overrides (
                sample_id TEXT PRIMARY KEY,
                work_group_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            );
            CREATE INDEX IF NOT EXISTS idx_annotation_review
                ON annotation_versions(review_status, tier);
            """
        )

    def save(self, record: AnnotationRecord, *, actor: str, action: str = "保存标注") -> None:
        """以乐观版本控制保存标注，并写入完整审计记录。"""
        record.validate()
        if not actor.strip():
            raise ConfigurationError("审核操作人不能为空。")
        payload = record.model_dump()
        with self.connection:
            existing = self.connection.execute(
                "SELECT raw_text FROM samples WHERE sample_id = ?", (record.sample_id,)
            ).fetchone()
            if existing is not None and existing["raw_text"] != record.raw_text:
                raise SchemaValidationError("样本ID已存在但原始标题不同。")
            self.connection.execute(
                "INSERT OR IGNORE INTO samples(sample_id, raw_text) VALUES (?, ?)",
                (record.sample_id, record.raw_text),
            )
            current = self.connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM annotation_versions WHERE sample_id = ?",
                (record.sample_id,),
            ).fetchone()["version"]
            if record.annotation.version != current + 1:
                raise SchemaValidationError("标注版本冲突，请重新加载最新版本后再保存。")
            self.connection.execute(
                """INSERT INTO annotation_versions(sample_id, version, tier, review_status, payload_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    record.sample_id,
                    record.annotation.version,
                    record.annotation.tier,
                    record.annotation.review_status,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            self.connection.executemany(
                """INSERT INTO spans(sample_id, version, position, start_offset, end_offset, label, text, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        record.sample_id,
                        record.annotation.version,
                        index,
                        span.start,
                        span.end,
                        span.label,
                        span.text,
                        span.source,
                    )
                    for index, span in enumerate(record.spans)
                ],
            )
            self.connection.execute(
                "INSERT INTO field_values(sample_id, version, extracted_json, catalog_json) VALUES (?, ?, ?, ?)",
                (
                    record.sample_id,
                    record.annotation.version,
                    json.dumps(asdict(record.extracted), ensure_ascii=False, sort_keys=True),
                    json.dumps(asdict(record.catalog), ensure_ascii=False, sort_keys=True)
                    if record.catalog
                    else None,
                ),
            )
            self.connection.execute(
                """INSERT INTO lineage(sample_id, version, sources_json, training_allowed, authorization_ref)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    record.sample_id,
                    record.annotation.version,
                    json.dumps(record.lineage.sources, ensure_ascii=False),
                    int(record.lineage.training_allowed),
                    record.lineage.authorization_ref,
                ),
            )
            self.connection.execute(
                "INSERT INTO audit_events(sample_id, version, actor, action) VALUES (?, ?, ?, ?)",
                (record.sample_id, record.annotation.version, actor, action),
            )

    def latest_payload(self, sample_id: str) -> dict[str, Any] | None:
        """读取一个样本的最新标注JSON，供审核界面恢复状态。"""
        row = self.connection.execute(
            """SELECT payload_json FROM annotation_versions
               WHERE sample_id = ? ORDER BY version DESC LIMIT 1""",
            (sample_id,),
        ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def latest_payloads(self) -> list[dict[str, Any]]:
        """读取每个样本的最新标注版本，供质量报告和导出使用。"""
        rows = self.connection.execute(
            """SELECT versions.payload_json FROM annotation_versions AS versions
               JOIN (
                   SELECT sample_id, MAX(version) AS version
                   FROM annotation_versions GROUP BY sample_id
               ) AS latest
                 ON versions.sample_id = latest.sample_id AND versions.version = latest.version
               ORDER BY versions.sample_id"""
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def review_queue(
        self,
        *,
        statuses: tuple[str, ...] = ("pending", "needs_review"),
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """只返回最新版本仍处于指定状态的审核队列记录。"""
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        rows = self.connection.execute(
            f"""SELECT versions.sample_id, versions.version, versions.tier, versions.review_status,
                       samples.raw_text
                FROM annotation_versions AS versions
                JOIN (
                    SELECT sample_id, MAX(version) AS version
                    FROM annotation_versions
                    GROUP BY sample_id
                ) AS latest
                  ON versions.sample_id = latest.sample_id AND versions.version = latest.version
                JOIN samples ON samples.sample_id = versions.sample_id
                WHERE versions.review_status IN ({placeholders})
                ORDER BY versions.created_at, versions.sample_id
                LIMIT ?""",
            (*statuses, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_sample_ids(self, *, limit: int = 100) -> list[str]:
        """返回最新版本仍待审核的样本ID，默认按保存时间排序。"""
        return [item["sample_id"] for item in self.review_queue(limit=limit)]

    def work_group_id(self, sample_id: str) -> str | None:
        """读取人工设置的作品组；未设置时返回空值。"""
        row = self.connection.execute(
            "SELECT work_group_id FROM work_group_overrides WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        return str(row["work_group_id"]) if row else None

    def work_group_overrides(self) -> dict[str, str]:
        """读取全部人工作品组覆盖，供数据划分批量使用。"""
        rows = self.connection.execute(
            "SELECT sample_id, work_group_id FROM work_group_overrides"
        ).fetchall()
        return {str(row["sample_id"]): str(row["work_group_id"]) for row in rows}

    def set_work_group(self, sample_id: str, work_group_id: str, *, actor: str) -> None:
        """保存人工作品组覆盖值，并记录关联到当前标注版本的审计事件。"""
        if not actor.strip() or not work_group_id.strip():
            raise ConfigurationError("作品组和操作人不能为空。")
        with self.connection:
            version_row = self.connection.execute(
                "SELECT MAX(version) AS version FROM annotation_versions WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            version = version_row["version"]
            if version is None:
                raise ConfigurationError("未找到需要设置作品组的样本。")
            self.connection.execute(
                """INSERT INTO work_group_overrides(sample_id, work_group_id, actor)
                   VALUES (?, ?, ?)
                   ON CONFLICT(sample_id) DO UPDATE SET
                       work_group_id = excluded.work_group_id,
                       actor = excluded.actor,
                       updated_at = CURRENT_TIMESTAMP""",
                (sample_id, work_group_id.strip(), actor.strip()),
            )
            self.connection.execute(
                "INSERT INTO audit_events(sample_id, version, actor, action) VALUES (?, ?, ?, ?)",
                (sample_id, version, actor.strip(), "人工设置作品组"),
            )
