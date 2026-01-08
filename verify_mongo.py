#!/usr/bin/env python3
"""Verify MongoDB storage - shows the latest forecast run."""

from utils.mongo_store import MongoForecastStore

def main():
    store = MongoForecastStore.from_env()
    
    # Get latest run
    runs = list(store.db.forecast_runs.find().sort("created_at", -1).limit(1))
    
    if runs:
        run = runs[0]
        print(f"Latest run: {run['_id']}")
        print(f"Created:    {run.get('created_at', 'N/A')}")
        print(f"Station:    {run.get('station', 'N/A')}")
        print(f"Members:    {run.get('members', 'N/A')}")
        
        # Count documents
        run_id = run["_id"]
        member_count = store.db.station_forecasts_member.count_documents({"run_id": run_id})
        final_count = store.db.station_forecasts_final.count_documents({"run_id": run_id})
        print(f"Member forecasts: {member_count}")
        print(f"Final forecasts:  {final_count}")
    else:
        print("No forecast runs found in MongoDB.")

if __name__ == "__main__":
    main()
