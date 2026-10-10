"""Step 6: plan delivery routes with Google OR-Tools.
It runs on made up coordinates as a small demo.
"""
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_vrp(distance_matrix, demands, capacities, depot=0):
    n_vehicles = len(capacities)
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix), n_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(distance_matrix[a][b])

    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_cb(i):
        return demands[manager.IndexToNode(i)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, capacities, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(params)
    return solution, routing, manager


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n_customers = 20
    coords = rng.uniform(0, 50, size=(n_customers + 1, 2))   # km, depot is row 0
    dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=2) * 100
    demands = [0] + [int(d) for d in rng.integers(1, 6, size=n_customers)]
    capacities = [20, 20, 20, 20]

    solution, routing, manager = solve_vrp(dist.tolist(), demands, capacities)

    # Baseline: one vehicle visits customers in the given order and returns
    baseline = dist[0][1] + sum(dist[i][i + 1] for i in range(1, n_customers))
    baseline += dist[n_customers][0]

    if solution is None:
        print("No solution found. Try more vehicles or capacity.")
    else:
        total = 0
        for v in range(len(capacities)):
            index = routing.Start(v)
            route, load = [], 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                load += demands[node]
                nxt = solution.Value(routing.NextVar(index))
                total += routing.GetArcCostForVehicle(index, nxt, v)
                index = nxt
            print(f"Vehicle {v}: route {route + [0]} load {load}")
        print(f"Optimised distance : {total / 100:.1f} km")
        print(f"Naive single route : {baseline / 100:.1f} km")
