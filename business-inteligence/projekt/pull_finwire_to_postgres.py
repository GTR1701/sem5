from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import psycopg


DEFAULT_API_BASE_URL = "https://public-api.finwire.pl"


@dataclass
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    sslmode: str | None = None


@dataclass
class ApiDataset:
    dataset_id: str
    endpoint: str


# Configure PostgreSQL in code.
POSTGRES_CONFIG = PostgresConfig(
    host="127.0.0.1",
    port=5432,
    dbname="finwire",
    user="postgres",
    password="zaq12wsx",
    sslmode=None,
)


# Configure direct API endpoints in code (one dataset = one table).
# Endpoints below are verified /v1 endpoints returning live JSON data.
API_DATASETS: list[ApiDataset] = [
    ApiDataset("snapshot", "/v1/snapshot"),
    ApiDataset("interest-rates", "/v1/series/interest-rate"),
    ApiDataset("cpi-monthly", "/v1/series/cpi?type=monthly&limit=600"),
    ApiDataset("cpi-yearly", "/v1/series/cpi?type=yearly"),
    ApiDataset("fx-table", "/v1/series/fx"),
    ApiDataset("public-debt", "/v1/public-debt"),
]


class ApiHttpClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 90,
        max_retries: int = 5,
        retry_base_delay_seconds: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds

    @staticmethod
    def _retry_after_seconds(exc: HTTPError) -> int | None:
        retry_after = exc.headers.get("Retry-After")
        if not retry_after:
            return None
        try:
            value = int(retry_after)
            return value if value >= 0 else None
        except ValueError:
            return None

    @staticmethod
    def _is_retryable_http(status_code: int, detail: str) -> bool:
        if status_code in {429, 500, 502, 503, 504}:
            return True
        if status_code == 403:
            body = detail.lower()
            return "error code: 1010" in body or "access denied" in body or "cloudflare" in body
        return False

    def get_json(self, endpoint: str) -> dict[str, Any]:
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        url = self.base_url + endpoint
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "finwire-api-etl/1.0 (+python urllib)",
            },
            method="GET",
        )
        raw: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                should_retry = attempt < self.max_retries and self._is_retryable_http(exc.code, detail)
                if not should_retry:
                    raise RuntimeError(f"HTTP {exc.code} for GET {endpoint}: {detail}") from exc

                retry_after = self._retry_after_seconds(exc)
                delay = retry_after if retry_after is not None else self.retry_base_delay_seconds * (
                    2**attempt
                )
                print(
                    f"Retryable HTTP {exc.code} for GET {endpoint}; "
                    f"attempt {attempt + 1}/{self.max_retries + 1}. Sleeping {delay:.1f}s."
                )
                time.sleep(delay)
            except URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Network error for GET {endpoint}: {exc}") from exc
                delay = self.retry_base_delay_seconds * (2**attempt)
                print(
                    f"Network error for GET {endpoint}; "
                    f"attempt {attempt + 1}/{self.max_retries + 1}. Sleeping {delay:.1f}s."
                )
                time.sleep(delay)

        if raw is None:
            raise RuntimeError(f"No response body for GET {endpoint} after retries")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from GET {endpoint}: {raw[:500]}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected response type from GET {endpoint}: {type(payload).__name__}")

        return payload


def build_conninfo(config: PostgresConfig) -> str:
    required = {
        "host": config.host,
        "port": config.port,
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
    }
    missing = [key for key, value in required.items() if value in ("", None)]
    if missing:
        raise RuntimeError(
            "Missing Postgres connection settings in POSTGRES_CONFIG: " + ", ".join(missing)
        )

    parts = [
        f"host={config.host}",
        f"port={config.port}",
        f"dbname={config.dbname}",
        f"user={config.user}",
        f"password={config.password}",
    ]
    if config.sslmode:
        parts.append(f"sslmode={config.sslmode}")
    return " ".join(parts)


def ensure_control_tables(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS finwire_dataset_catalog (
                dataset_id TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                table_name TEXT NOT NULL,
                last_pulled_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS finwire_pull_errors (
                id BIGSERIAL PRIMARY KEY,
                run_id UUID NOT NULL,
                dataset_id TEXT,
                endpoint TEXT,
                occurred_at TIMESTAMPTZ NOT NULL,
                error_message TEXT NOT NULL
            );
            """
        )
    conn.commit()


def jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def dataset_table_name(dataset_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", dataset_id.lower()).strip("_")
    if not normalized:
        normalized = "dataset"
    base = f"finwire_ds_{normalized}"
    if len(base) <= 63:
        return base
    suffix = hashlib.sha1(dataset_id.encode("utf-8")).hexdigest()[:8]
    return f"{base[:54]}_{suffix}"


def index_name(name_base: str) -> str:
    if len(name_base) <= 63:
        return name_base
    suffix = hashlib.sha1(name_base.encode("utf-8")).hexdigest()[:8]
    return f"{name_base[:54]}_{suffix}"


BASE_DATASET_COLUMNS = {
    "id",
    "run_id",
    "pulled_at",
    "dataset_id",
    "endpoint",
    "record_index",
    "record_key",
    "record_id",
    "record_code",
    "record_date",
    "record_period",
    "record_year",
    "record_quarter",
    "record",
    "dataset_meta",
    "response_info",
}


def sanitize_column_name(raw: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if not normalized:
        normalized = "field"
    col = f"f_{normalized}"
    if len(col) <= 63:
        return col
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{col[:54]}_{suffix}"


def value_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def flatten_record_values(
    value: Any,
    path: str,
    out: dict[str, str | None],
    used_columns: set[str],
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_path = f"{path}_{key}" if path else str(key)
            flatten_record_values(nested, next_path, out, used_columns)
        return

    column = sanitize_column_name(path if path else "value")
    if column in used_columns or column in BASE_DATASET_COLUMNS:
        suffix = hashlib.sha1(path.encode("utf-8")).hexdigest()[:6]
        column = f"{column[:56]}_{suffix}" if len(column) > 56 else f"{column}_{suffix}"
    used_columns.add(column)
    out[column] = value_to_text(value)


def record_to_columns(record: dict[str, Any]) -> dict[str, str | None]:
    columns: dict[str, str | None] = {}
    used: set[str] = set()
    for key, value in record.items():
        flatten_record_values(value, str(key), columns, used)
    return columns


def get_table_columns(conn: psycopg.Connection[Any], table_name: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s;
            """,
            (table_name,),
        )
        rows = cur.fetchall()
    return {str(row[0]) for row in rows}


def ensure_record_columns(
    conn: psycopg.Connection[Any],
    table_name: str,
    record_columns: set[str],
    existing_columns: set[str],
) -> None:
    from psycopg import sql

    missing = [col for col in record_columns if col not in existing_columns]
    if not missing:
        return

    with conn.cursor() as cur:
        for col in missing:
            cur.execute(
                sql.SQL("ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col} TEXT;").format(
                    table_name=sql.Identifier(table_name),
                    col=sql.Identifier(col),
                )
            )
            existing_columns.add(col)


def ensure_dataset_table(conn: psycopg.Connection[Any], table_name: str) -> None:
    from psycopg import sql

    create_sql = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id BIGSERIAL PRIMARY KEY,
            run_id UUID NOT NULL,
            pulled_at TIMESTAMPTZ NOT NULL,
            dataset_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            record_index INTEGER NOT NULL,
            record_key TEXT,
            record_id TEXT,
            record_code TEXT,
            record_date TEXT,
            record_period TEXT,
            record_year INTEGER,
            record_quarter TEXT,
            record JSONB NOT NULL,
            dataset_meta JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            response_info JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        """
    ).format(table_name=sql.Identifier(table_name))

    idx_1 = sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table_name} (pulled_at DESC);").format(
        idx=sql.Identifier(index_name(f"{table_name}_pulled_at_idx")),
        table_name=sql.Identifier(table_name),
    )
    idx_2 = sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table_name} (record_date, record_period);").format(
        idx=sql.Identifier(index_name(f"{table_name}_record_time_idx")),
        table_name=sql.Identifier(table_name),
    )

    with conn.cursor() as cur:
        cur.execute(create_sql)
        cur.execute(idx_1)
        cur.execute(idx_2)
    conn.commit()


def iter_records(data: Any) -> list[tuple[int, str | None, Any]]:
    records: list[tuple[int, str | None, Any]] = []
    idx = 0

    def push(record_key: str | None, value: Any) -> None:
        nonlocal idx
        records.append((idx, record_key, value))
        idx += 1

    if isinstance(data, dict):
        if isinstance(data.get("entries"), list):
            for item in data["entries"]:
                push("entries", item)
        elif isinstance(data.get("points"), list):
            for item in data["points"]:
                push("points", item)
        else:
            found_list = False
            for key, value in data.items():
                if isinstance(value, list):
                    found_list = True
                    for item in value:
                        push(key, item)
            if not found_list:
                push(None, data)
    elif isinstance(data, list):
        for item in data:
            push(None, item)
    else:
        push(None, data)

    return records


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def insert_dataset_records(
    conn: psycopg.Connection[Any],
    table_name: str,
    run_id: uuid.UUID,
    pulled_at: datetime,
    dataset_id: str,
    endpoint: str,
    payload: dict[str, Any],
) -> int:
    from psycopg import sql

    data_block = payload.get("data")
    meta_block = payload.get("meta", {})
    if not isinstance(meta_block, dict):
        meta_block = {"value": meta_block}

    response_info = {k: v for k, v in payload.items() if k not in {"data", "meta"}}
    records = iter_records(data_block)
    if not records:
        records = [(0, None, data_block)]

    inserted = 0
    existing_columns = get_table_columns(conn, table_name)
    with conn.cursor() as cur:
        for record_index, record_key, record in records:
            rec_dict = record if isinstance(record, dict) else {"value": record}

            dynamic_columns = record_to_columns(rec_dict)
            ensure_record_columns(conn, table_name, set(dynamic_columns.keys()), existing_columns)

            base_columns = [
                "run_id",
                "pulled_at",
                "dataset_id",
                "endpoint",
                "record_index",
                "record_key",
                "record_id",
                "record_code",
                "record_date",
                "record_period",
                "record_year",
                "record_quarter",
                "record",
                "dataset_meta",
                "response_info",
            ]
            base_values: list[Any] = [
                run_id,
                pulled_at,
                dataset_id,
                endpoint,
                record_index,
                record_key,
                str(rec_dict.get("id")) if rec_dict.get("id") is not None else None,
                str(rec_dict.get("code")) if rec_dict.get("code") is not None else None,
                str(rec_dict.get("date")) if rec_dict.get("date") is not None else None,
                str(rec_dict.get("period")) if rec_dict.get("period") is not None else None,
                to_int(rec_dict.get("year")),
                str(rec_dict.get("quarter")) if rec_dict.get("quarter") is not None else None,
                jsonb(rec_dict),
                jsonb(meta_block),
                jsonb(response_info),
            ]

            dynamic_column_names = sorted(dynamic_columns.keys())
            dynamic_values = [dynamic_columns[name] for name in dynamic_column_names]

            all_columns = base_columns + dynamic_column_names
            all_values = base_values + dynamic_values

            insert_sql = sql.SQL("INSERT INTO {table_name} ({columns}) VALUES ({values});").format(
                table_name=sql.Identifier(table_name),
                columns=sql.SQL(", ").join(sql.Identifier(c) for c in all_columns),
                values=sql.SQL(", ").join(sql.Placeholder() for _ in all_columns),
            )
            cur.execute(insert_sql, all_values)
            inserted += 1
    conn.commit()
    return inserted


def upsert_catalog(
    conn: psycopg.Connection[Any],
    dataset_id: str,
    endpoint: str,
    table_name: str,
    pulled_at: datetime | None,
) -> None:
    now = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO finwire_dataset_catalog (dataset_id, endpoint, table_name, last_pulled_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (dataset_id) DO UPDATE SET
                endpoint = EXCLUDED.endpoint,
                table_name = EXCLUDED.table_name,
                last_pulled_at = EXCLUDED.last_pulled_at,
                updated_at = EXCLUDED.updated_at;
            """,
            (dataset_id, endpoint, table_name, pulled_at, now),
        )
    conn.commit()


def insert_pull_error(
    conn: psycopg.Connection[Any],
    run_id: uuid.UUID,
    dataset_id: str | None,
    endpoint: str | None,
    message: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO finwire_pull_errors (run_id, dataset_id, endpoint, occurred_at, error_message)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (run_id, dataset_id, endpoint, datetime.now(timezone.utc), message),
        )
    conn.commit()


def pull_and_store_all(
    conn: psycopg.Connection[Any],
    client: ApiHttpClient,
    datasets: list[ApiDataset],
    run_id: uuid.UUID,
    delay_seconds: float,
) -> tuple[int, int]:
    success_count = 0
    error_count = 0

    for index, ds in enumerate(datasets, start=1):
        table_name = dataset_table_name(ds.dataset_id)
        print(f"[{index}/{len(datasets)}] GET {ds.endpoint} -> {table_name}")
        try:
            payload = client.get_json(ds.endpoint)
            pulled_at = datetime.now(timezone.utc)
            ensure_dataset_table(conn, table_name)
            inserted = insert_dataset_records(
                conn=conn,
                table_name=table_name,
                run_id=run_id,
                pulled_at=pulled_at,
                dataset_id=ds.dataset_id,
                endpoint=ds.endpoint,
                payload=payload,
            )
            upsert_catalog(conn, ds.dataset_id, ds.endpoint, table_name, pulled_at)
            print(f"  saved rows: {inserted}")
            success_count += 1
        except Exception as exc:
            error_count += 1
            message = str(exc)
            print(f"  ERROR: {message}")
            insert_pull_error(conn, run_id, ds.dataset_id, ds.endpoint, message)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return success_count, error_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull live data from finwire /v1 API and store one dataset per PostgreSQL table."
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Base API URL (default: https://public-api.finwire.pl)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Delay between dataset pulls (default: 0.5)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="HTTP timeout for one API request (default: 90)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for retryable HTTP/network errors (default: 5)",
    )
    parser.add_argument(
        "--retry-base-delay-seconds",
        type=float,
        default=1.5,
        help="Base exponential backoff delay in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of configured datasets to pull (0 = all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import psycopg

    datasets = API_DATASETS[: args.limit] if args.limit > 0 else API_DATASETS
    if not datasets:
        raise RuntimeError("No datasets configured in API_DATASETS.")

    conninfo = build_conninfo(POSTGRES_CONFIG)
    client = ApiHttpClient(
        base_url=args.api_base_url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_base_delay_seconds=args.retry_base_delay_seconds,
    )
    run_id = uuid.uuid4()

    print(f"Run ID: {run_id}")
    print(f"Datasets to pull: {len(datasets)}")

    with psycopg.connect(conninfo) as conn:
        ensure_control_tables(conn)
        ok, failed = pull_and_store_all(
            conn=conn,
            client=client,
            datasets=datasets,
            run_id=run_id,
            delay_seconds=args.delay_seconds,
        )

    print(f"Completed. Success: {ok}, Failed: {failed}, Run ID: {run_id}")
    if ok == 0 and failed > 0:
        raise RuntimeError("No data rows were saved. Check finwire_pull_errors.")


if __name__ == "__main__":
    main()
