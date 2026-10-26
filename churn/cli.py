from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import select

from churn.db import Customer, Organization, session_factory
from churn.migrate import LATEST, current_version, init_schema, migrate, pending
from churn.ml.contract import SOURCE_KINDS
from churn.service import import_dataset, latest_model, register_model, score_customers


def main() -> None:
    parser = argparse.ArgumentParser(description="Local administrator commands")
    parser.add_argument("command", choices=["init-db", "migrate", "import", "register", "score-all"])
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--organization", default="Portfolio demo")
    parser.add_argument("--source-kind", choices=SOURCE_KINDS,
                        help="required for import: declares whether the file is synthetic or external")
    parser.add_argument("--check", action="store_true", help="migrate: report pending migrations only")
    args = parser.parse_args()
    factory, engine = session_factory()
    if args.command == "init-db":
        init_schema(engine)
        print(f"Database initialized at schema version {LATEST}")
        return
    if args.command == "migrate":
        before = current_version(engine)
        if before is None:
            parser.error("Database is empty; use init-db")
        todo = pending(engine)
        if args.check:
            print(f"Schema version {before}; latest {LATEST}; pending: "
                  f"{[m.name for m in todo] or 'none'}")
            raise SystemExit(1 if todo else 0)
        applied, backup = migrate(engine)
        if backup:
            print(f"SQLite backup written to {backup}")
        elif applied:
            print("WARNING: no automatic backup for this database engine; ensure you have one")
        print(f"Applied: {[m.name for m in applied] or 'nothing (already current)'}")
        print("Rows without recorded lineage remain NULL and are reported as unverified.")
        return
    with factory() as session:
        if args.command == "import":
            if not args.path:
                parser.error("import requires a CSV path")
            if not args.source_kind:
                parser.error("import requires --source-kind synthetic|external")
            item = import_dataset(session, args.path, args.organization, args.source_kind)
            print(f"Imported dataset {item.id}: {item.rows} rows, {item.source_kind}, "
                  f"{item.contract_version}, quality {item.quality_status}")
        elif args.command == "register":
            if not args.path:
                parser.error("register requires a report JSON path")
            model = register_model(session, args.path, args.organization)
            print(f"Registered candidate {model.id}")
        elif args.command == "score-all":
            organization = session.scalar(select(Organization).where(Organization.name == args.organization))
            if organization is None:
                parser.error("Unknown organization")
            model = latest_model(session, organization.id)
            if model is None:
                parser.error("No registered model")
            customers = session.scalars(select(Customer).where(
                Customer.organization_id == organization.id, Customer.dataset_id == model.dataset_id)).all()
            try:
                count = score_customers(session, list(customers), model)
            except ValueError as exc:
                parser.error(str(exc))
            print(f"Scored {count} customers with {model.id} (retrospective batch scores; training "
                  "and calibration customers are in-sample, see split_assignments)")


if __name__ == "__main__":
    main()
