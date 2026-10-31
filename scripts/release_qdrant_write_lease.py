"""Release an abandoned Qdrant write lease after the collection has been reconciled.

An expired ``ledger_write_lease`` makes every API instance fail closed. Run this
only after the authoritative document/ACL state has been re-ingested or the
collection rebuilt and validated. It reads the same ``LEDGER_*`` settings as the
API (including ``LEDGER_SECRETS_DIR``) and changes only that metadata key.
"""

import argparse
import asyncio

from ledger_rag.core.config import get_settings
from ledger_rag.retrieval.qdrant import QdrantRetriever


async def release() -> None:
    store = QdrantRetriever(get_settings())
    try:
        await store.release_write_lease()
        print(f"released write lease on collection {get_settings().qdrant_collection}")
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-reconciled",
        action="store_true",
        help="Acknowledge that the collection was reconciled with the source of truth",
    )
    if not parser.parse_args().confirm_reconciled:
        parser.error("refusing to release the lease without --confirm-reconciled")
    asyncio.run(release())


if __name__ == "__main__":
    main()
