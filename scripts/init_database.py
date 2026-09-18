from __future__ import annotations

from relevanceflow.database.session import Base, create_database_engine


def main() -> None:
    """Create RelevanceFlow database tables."""
    engine = create_database_engine()

    print("Creating database tables...")

    Base.metadata.create_all(engine)

    print("Database tables created successfully.")

    engine.dispose()


if __name__ == "__main__":
    main()
