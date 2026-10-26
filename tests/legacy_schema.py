"""Schema DDL of the databases created by earlier phases (copied from real files)."""

from __future__ import annotations

V0_DDL = '''
CREATE TABLE organizations (
	id VARCHAR(36) NOT NULL, 
	name VARCHAR(120) NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (name)
);
CREATE TABLE datasets (
	id VARCHAR(36) NOT NULL, 
	organization_id VARCHAR(36) NOT NULL, 
	definition_key VARCHAR(120) NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	rows INTEGER NOT NULL, 
	source_kind VARCHAR(20) NOT NULL, 
	imported_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (organization_id, sha256), 
	FOREIGN KEY(organization_id) REFERENCES organizations (id)
);
CREATE INDEX ix_datasets_organization_id ON datasets (organization_id);
CREATE TABLE customers (
	id VARCHAR(36) NOT NULL, 
	organization_id VARCHAR(36) NOT NULL, 
	external_id VARCHAR(120) NOT NULL, 
	dataset_id VARCHAR(36) NOT NULL, 
	attributes JSON NOT NULL, 
	observed_exit INTEGER, 
	PRIMARY KEY (id), 
	UNIQUE (organization_id, external_id), 
	FOREIGN KEY(organization_id) REFERENCES organizations (id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (id)
);
CREATE INDEX ix_customers_organization_id ON customers (organization_id);
CREATE TABLE model_versions (
	id VARCHAR(120) NOT NULL, 
	dataset_id VARCHAR(36) NOT NULL, 
	definition_key VARCHAR(120) NOT NULL, 
	artifact_path VARCHAR(500) NOT NULL, 
	metrics JSON NOT NULL, 
	trained_at DATETIME NOT NULL, 
	status VARCHAR(20) NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (id)
);
CREATE TABLE predictions (
	id VARCHAR(36) NOT NULL, 
	customer_id VARCHAR(36) NOT NULL, 
	model_id VARCHAR(120) NOT NULL, 
	probability FLOAT NOT NULL, 
	risk_band VARCHAR(16) NOT NULL, 
	scored_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(customer_id) REFERENCES customers (id), 
	FOREIGN KEY(model_id) REFERENCES model_versions (id)
);
CREATE INDEX ix_predictions_customer_id ON predictions (customer_id);
'''

# Phase 2 added these columns; a v1 database is a v0 database plus them.
V1_ALTERS = [
    "ALTER TABLE datasets ADD COLUMN contract_version VARCHAR(80) NOT NULL DEFAULT 'banking-csv-contract/1.0.0'",
    "ALTER TABLE datasets ADD COLUMN quality_status VARCHAR(30) NOT NULL DEFAULT 'passed'",
    "ALTER TABLE datasets ADD COLUMN quality_report_sha256 VARCHAR(64) NOT NULL DEFAULT 'a'",
    "ALTER TABLE datasets ADD COLUMN quality_report JSON NOT NULL DEFAULT '{\"issues\": [], \"status\": \"passed\"}'",
    "ALTER TABLE model_versions ADD COLUMN artifact_sha256 VARCHAR(64) NOT NULL DEFAULT 'b'",
    "ALTER TABLE model_versions ADD COLUMN lineage JSON NOT NULL DEFAULT '{\"dataset\": {}}'",
]
