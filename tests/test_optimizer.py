from src.optimizer import drone_feasibility, first_fit_payload, nearest_neighbor_route

def test_drone_constraints():
    result=drone_feasibility(12,2,20,3,100,20)
    assert result["feasible"]
    assert not drone_feasibility(18,2,20,3,100,20)["feasible"]

def test_route_and_payload():
    route=nearest_neighbor_route((12.97,77.59),[{"id":"A","lat":12.98,"lon":77.60},{"id":"B","lat":13.00,"lon":77.62}])
    assert len(route["ordered_stops"])==2 and route["total_distance_km"]>0
    trips=first_fit_payload([{"id":1,"weight_kg":2},{"id":2,"weight_kg":2}],3)
    assert len(trips)==2

