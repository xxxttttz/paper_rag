"""Start the RQ worker used for knowledge-base ingestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rq import Worker
from rq.serializers import JSONSerializer

from job_queue import ingestion_queue, redis_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Process the current queue and exit when it becomes empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = redis_connection()
    connection.ping()
    queue = ingestion_queue(connection)
    worker = Worker(
        [queue],
        connection=connection,
        serializer=JSONSerializer,
    )
    worker.work(burst=args.burst)


if __name__ == "__main__":
    main()
