import argparse
from processor.es_indexer import ESIndexer


def main():
    parser = argparse.ArgumentParser(description="Build Elasticsearch index for chunks")
    parser.add_argument("--recreate", action="store_true", help="Recreate index")
    parser.add_argument("--from-path", default=None, help="Source path (default config FILES_DIR)")
    parser.add_argument("--batch-size", type=int, default=500, help="Bulk batch size")
    args = parser.parse_args()

    indexer = ESIndexer()
    mode = indexer.create_index(recreate=args.recreate)
    print(f"Index ready. analyzer={mode}")
    stats = indexer.index_from_path(from_path=args.from_path, batch_size=args.batch_size)
    print(f"Indexed: ok={stats['ok']} fail={stats['fail']} took_ms={stats['took_ms']} last_error={stats['last_error']}")


if __name__ == "__main__":
    main()


