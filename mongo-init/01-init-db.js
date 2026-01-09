// MongoDB initialization script for FuXi-S2S forecasts
// This script runs automatically when the MongoDB container starts for the first time

// Switch to the arice database
db = db.getSiblingDB("arice");

// Create collections with validation
db.createCollection("forecasts", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["init_date", "station", "created_at"],
      properties: {
        init_date: {
          bsonType: "string",
          description: "Initialization date in YYYYMMDD format",
        },
        station: {
          bsonType: "string",
          description: "Station name for the forecast",
        },
        members: {
          bsonType: "int",
          description: "Number of ensemble members",
        },
        forecast_data: {
          bsonType: "object",
          description: "Forecast data object",
        },
        created_at: {
          bsonType: "date",
          description: "Timestamp when the forecast was stored",
        },
      },
    },
  },
});

// Create indexes for efficient querying
db.forecasts.createIndex({ init_date: 1 });
db.forecasts.createIndex({ station: 1 });
db.forecasts.createIndex({ init_date: 1, station: 1 }, { unique: true });
db.forecasts.createIndex({ created_at: -1 });

// Create metadata collection
db.createCollection("metadata");
db.metadata.insertOne({
  type: "schema_version",
  version: "1.0.0",
  created_at: new Date(),
  description: "FuXi-S2S forecast storage schema",
});

print("✓ FuXi-S2S MongoDB initialized successfully");
print("  - Created forecasts collection with validation");
print("  - Created indexes for efficient querying");
