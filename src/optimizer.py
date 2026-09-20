from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt


def haversine_km(a: tuple[float,float], b: tuple[float,float]) -> float:
    lat1,lon1,lat2,lon2=map(radians,[a[0],a[1],b[0],b[1]])
    dlat,dlon=lat2-lat1,lon2-lon1
    h=sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371*2*asin(sqrt(h))


def nearest_neighbor_route(origin: tuple[float,float], stops: list[dict], return_to_origin: bool=True) -> dict:
    remaining=[dict(x) for x in stops]; route=[]; current=origin; total=0.0
    while remaining:
        nxt=min(remaining,key=lambda x:haversine_km(current,(x["lat"],x["lon"])))
        distance=haversine_km(current,(nxt["lat"],nxt["lon"])); total+=distance
        route.append({**nxt,"leg_distance_km":round(distance,2)}); current=(nxt["lat"],nxt["lon"]); remaining.remove(nxt)
    if return_to_origin and route: total+=haversine_km(current,origin)
    return {"ordered_stops":route,"total_distance_km":round(total,2),"returns_to_origin":return_to_origin}


def drone_feasibility(distance_km: float, payload_kg: float, max_range_km: float, capacity_kg: float, battery_start_pct: float=100, reserve_pct: float=20) -> dict:
    usable_range=max_range_km*max(0,battery_start_pct-reserve_pct)/100
    reasons=[]
    if payload_kg>capacity_kg: reasons.append("payload exceeds capacity")
    if distance_km>usable_range: reasons.append("distance exceeds usable battery range")
    return {"feasible":not reasons,"usable_range_km":round(usable_range,2),"reasons":reasons}


def first_fit_payload(items: list[dict], capacity_kg: float) -> list[list[dict]]:
    trips=[]
    for item in sorted(items,key=lambda x:x["weight_kg"],reverse=True):
        for trip in trips:
            if sum(x["weight_kg"] for x in trip)+item["weight_kg"]<=capacity_kg: trip.append(item); break
        else: trips.append([item])
    return trips


if __name__ == "__main__":
    sample_origin = (12.9716, 77.5946)
    sample_stops = [
        {"name": "Customer A", "lat": 12.9352, "lon": 77.6245},
        {"name": "Customer B", "lat": 12.9698, "lon": 77.7500},
        {"name": "Customer C", "lat": 13.0358, "lon": 77.5970},
    ]

    route = nearest_neighbor_route(sample_origin, sample_stops)
    print("SmartLogix route-optimization test")
    print("Optimized stop order:")
    for number, stop in enumerate(route["ordered_stops"], start=1):
        print(f"  {number}. {stop['name']} ({stop['leg_distance_km']} km)")
    print(f"Total round-trip distance: {route['total_distance_km']} km")

    feasibility = drone_feasibility(
        distance_km=10,
        payload_kg=2,
        max_range_km=20,
        capacity_kg=5,
    )
    print(f"Drone delivery feasible: {feasibility['feasible']}")
    print(f"Usable drone range: {feasibility['usable_range_km']} km")
