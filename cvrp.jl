### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 7d199918-a41c-4434-b726-3cc770a67667
begin
	using CpuId
	cpuinfo()
end

# ╔═╡ 26e41c3f-d6a1-4b71-a01c-09f0e6473dff
begin
	using Pipe
	
	using JuMP
	using HiGHS

	using LinearAlgebra

	using DataFrames
	using CSV
	using Dates
end

# ╔═╡ 0e27cbaa-6593-46e5-8d3d-1bd34174fc8a
begin
	using Downloads
	
	"""
	    download_cvrp_instances(data_dir::String)
	
	Download a set of CVRP instance files to a specified directory.
	
	# Arguments
	- `data_dir::String`: Path to the directory where the CVRP instance files will be saved.
	
	# Description
	This function creates the specified directory if it does not exist and downloads a predefined set 
	of CVRP instance files from a remote server to this directory.
	
	# Example
	```julia
	download_cvrp_instances("path/to/data_dir")
	```
	
	# Notes
	- The list of CVRP instances is hardcoded within the function.
	- The instances are downloaded from `http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/E/`.
	"""
	function download_cvrp_instances(data_dir)
	    # Create data directory if it doesn't exist
	    if !isdir(data_dir)
	        mkdir(data_dir)
	    end
	
	    # List of instances
	    instances = [
	        "E-n13-k4", "E-n22-k4", "E-n23-k3", "E-n30-k3", "E-n31-k7", "E-n33-k4",
	        "E-n51-k5", "E-n76-k7", "E-n76-k8", "E-n76-k10", "E-n76-k14", "E-n101-k8", "E-n101-k14"
	    ]
	
	    # Download each instance file
	    for instance in instances
	        url = "http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/E/$(instance).vrp"
	        file_path = joinpath(data_dir, "$(instance).vrp")
	        Downloads.download(url, file_path)
	    end
	end

end

# ╔═╡ 0bb30050-dc8d-11ee-0b8c-8d66b5044644
md"""
# The Capacitated Vehicle Routing Problem (CVRP)
"""

# ╔═╡ 68564574-d1b2-470c-ab4e-0c221bc14895
md"""
> A fleet of identical vehicles, with limited capacity, is located at a depot. There are n customers that require service. Each customer has a known demand. The cost of travel between any pair of customers, or between any customer and the depot, is also known. The task is to find a minimum-cost collection of vehicle routes, each starting and ending at the depot, such that each customer is visited by exactly one vehicle, and no vehicle visits a set of customers whose total demand exceeds the vehicle capacity.
"""

# ╔═╡ 41796f80-e24e-4cbd-aa68-d1e608fa9a4f
md"""
We are given:
- A depot,
- A fleet of identical vehicles with capacity $Q$,
- n customers with demands $q_i \le Q$, for $i = 1, \ldots, n$, to be delivered from the depot,
- Symmetric travel cost $c_{i,j}$ between points $i$ and $j$ for all pairs of points (i.e., the customers and the depot).

The goal is to determine routes of minimum total travel cost, subject to the following constraints:
- Each customer is serviced exactly once and she gets the desired demand,
- The total quantity delivered on each route does not exceed $Q$,
- Each route begins and ends at the depot.
"""

# ╔═╡ 0edb3e3a-5f3d-4148-9c13-11b72b3ceef6
data_dir = "data/"

# ╔═╡ 08916fda-f183-4867-b62e-e9efc41fe4de
# Uncomment to download CVRP instances
# download_cvrp_instances(data_dir)

# ╔═╡ 7a487842-5fc4-4504-a7bb-4b3525f84e8a
# Internal representation of the instance
@kwdef struct CVRP
	name :: String
	dimension :: Int
	optimal_value :: Float64
	capacity :: Float64
	C :: Matrix{Float64}
	q :: Vector{Float64}
end

# ╔═╡ 1ca827f2-9c53-448f-abb2-961c080c7b6f
md"""
NOTE: We assert that depot has index 1 and customers $\in \{2 \cdots n\}$.
"""

# ╔═╡ 5a38f991-19a2-47b3-9149-a6b650fcea76
md"""
## ILP Formulation
"""

# ╔═╡ 7c250684-5899-4cc7-a876-053130e13068
md"""
In this section we describe the ILP formulation of the problem.
"""

# ╔═╡ 8527a61c-5899-47f7-bf32-bc66dff5ae16
md"""
Firstly, we start by defining binary variables $x_{ij}$ for each arc with the following meaning:

$$\begin{equation*}
x_{ij} \sim \text{ vehicle traverses from node $i$ to node $j$}
\end{equation*}$$

Using these variables we define

- Kirchhoff's law (in deegree is the same as out degree)

$$\begin{align}
\sum_{j} x_{ij} = \sum_{j} x_{ji} && \forall\; i \in 1, \ldots, n
\end{align}$$

- every customer is visited once

$$\begin{align}
\sum_{j} x_{ij} = 1 && \forall\; j \in 2, \ldots, n
\end{align}$$

- binary constraint $x_{ij} \in \{0, 1\}$

These constraints by themselves are not strict enough, as they allow formation of circulations or subtours and we also do not limit the quantity delivered on a route. We solve both of these problems by assigning a number $y_i$ to each _customer_ in such a way that if a vehicle follows a path $i \to j$ then $y_j > y_i$.

Notice that in order to be satisfied, no cycles can be formed besides those running through the depot.

Moreover, $y_i$ will also represent the running quantity delivered by a vehicle on a path from the depot to $i$. By enforcing all $y_i$ to be at most $Q$, we satisfy the vehicle limits.

We add the (real) variable $y_i$ for all $i = 2, \ldots, n$

$$\begin{equation*}
y_{i} \sim \text{ the running amount delivered by a vehicle on a path from depot to $i$}
\end{equation*}$$

- limit the quantity

$$\begin{align}
	q_i \le y_i \le Q && \forall\; i \in 2, \ldots, n
\end{align}$$

- enforce increase by the required quantity of a customer

$$\begin{align}
	y_j - y_i \ge q_j- Q(1 - x_{ij})
\end{align}$$

To verify the correctness of this formulation consider cases

1. The arc $ij$ is _used_ (i. e. $x_{ij} = 1$). Then the condition reads $y_j - y_i \ge q_j$ and $y_j$ is forced to be increased by at least $q_j$ with respect to $i$.

2. The arc $ij$ is _unused_ (i. e. $x_{ij} = 1$). Then the condition can be formulated as $y_j - q_j \ge y_i - Q$ which always trivially holds thanks to the variable bounds.
"""

# ╔═╡ 4d052081-11dd-4780-835b-f8601ceb53ab
md"""
The conditions are translated to code below using the [JuMP](https://jump.dev/) library.
"""

# ╔═╡ dd8c3dd5-31e0-4549-8e4f-d1d5eaf11d6f
function formulate(instance :: CVRP, model :: Model; relax=false)
	n = instance.dimension
	
	nodes = 1:n

	C = instance.C
	Q = instance.capacity
	q = instance.q

	# Variable definitions with bounds
	@variable(model, x[i=nodes, j=nodes; i != j], Bin)
	@variable(model, y[i=2:n])

	# Kirchoff law
	@constraint(model, [j=nodes], 
		sum(x[:, j]) == sum(x[j, :])
	)

	# Every node entered once
	@constraint(model, [j = 2:n ],
		sum(x[:, j]) == 1
	)

	# Capacity constraints and no subtours
	@constraint(model, [i = 2:n, j = 2:n; i != j],
		y[j] - y[i] >= q[j] - Q * (1 - x[i, j])
	)
	@constraint(model, [i = 2:n],
		q[i] <= y[i] <= Q
	)
	
	
	@objective(
        model,
        Min,
        sum(
			C[i, j] * x[i, j]
			for i ∈ nodes, j ∈ nodes
			if i != j
		)
    )

	model, (x, y)
end

# ╔═╡ 3b3aed4f-cd75-41fd-bdb4-50bf1a7daefb
md"""
# Resuts
"""

# ╔═╡ 2511a635-4115-43a0-8c66-96cf48990b08
md"""
In this section, we present the results obtained from solving the ILP model for out problem instances.
"""

# ╔═╡ 3eac7f08-6d6e-40aa-a939-97fe1d4e58e7
md"""
### System information
"""

# ╔═╡ 732bf513-f9c5-4fc8-a89f-f1f99f26cb31
md"""
The computational experiments were conducted on a machine with the following specifications:
"""

# ╔═╡ fd7939f0-a2cf-4f16-abee-257031e67696
md"""
- OS: $(Sys.KERNEL)
- Memory: $(Sys.total_memory() / 2^30 |> round) GB
"""

# ╔═╡ 3a43af06-e283-4fa4-928a-5c2d6c00b669
md"""
## Software

For solving the ILP model, we used the JuMP package in Julia, coupled with the HiGHS solver.
"""

# ╔═╡ bbf736fe-6d1d-4030-a625-7be6ddd03690
# Time limit in seconds for solver
TIMEOUT = 1200 # 20 minutes

# ╔═╡ 0f47a17f-7d8f-47e8-ba93-505e86de8bcf
md"""
## Summary of Results

The table below summarizes the performance and outcomes of the ILP model across different problem instances. For each instance, we provide the following details:

- Name of the instance
- Size of the instance (number of customers)
- LP OPT (Optimal value for the linear programming relaxation)
- ILP OPT (Optimal value for the integer linear programming model)
- Time for obtaining LP OPT (in seconds)
- Time for obtaining ILP OPT (in seconds)
- Best known upper bound on the optimum
- Best known lower bound on the optimum
- Relative gap
"""

# ╔═╡ 2646ca7f-99e2-4bab-abde-d292b99a86f9
md"""
---

# Helper functions
"""

# ╔═╡ b72bde0e-00ee-4a51-8ab5-6463de37a260
"""
    load_cvrp(instance_file::String) -> CVRP

Load a CVRP instance from a file.

# Arguments
- `instance_file::String`: Path to the CVRP instance file.

# Returns
- `CVRP`: A CVRP object with the loaded instance data.

# Description
Parses a CVRP instance file to extract the problem name, optimal value, dimension, vehicle capacity, 
distance matrix (either explicit or Euclidean), demands, and depot information.

# Example
```julia
cvrp_instance = load_cvrp("path/to/instance/file.vrp")
```

# Errors
- Throws an `ArgumentError` for an invalid edge weight type.
"""
function load_cvrp(instance_file :: String)
	instance_file = open(instance_file)

	_, name = @pipe readline(instance_file) |> split(_, " : ")
	_, comment = @pipe readline(instance_file) |> split(_, " : ")
	_, typ = @pipe readline(instance_file) |> split(_, " : ")
	_, dimension = @pipe readline(instance_file) |> split(_, " : ")
	_, edge_weight_type = @pipe readline(instance_file) |> split(_, " : ")
	if edge_weight_type == "EXPLICIT"
		readline(instance_file) # EDGE_WEIGHT_FORMAT
		readline(instance_file) # DISPLAY_DATA_TYPE
	end
	_, capacity = @pipe readline(instance_file) |> split(_, " : ")

	r = Regex("value: (\\d+)")
	m = match(r, comment)
	optimal_value = parse(Float64, m.captures[1])

	Q = parse(Float64, capacity)
	n = parse(Int, dimension)
	D = zeros(Float64, n, n)
	if edge_weight_type == "EXPLICIT"
		readline(instance_file) # EDGE_WEIGHT_SECTION
		
		local i, j = 2, 1
		
		local line = readline(instance_file)
		while !contains(line, "DEMAND_SECTION")
			for num_str in split(line)
				d = parse(Float64, num_str)
				D[i, j] = d
				D[j, i] = d

				if i == j + 1
					i += 1
					j = 1
				else
					j += 1
				end
			end
			line = readline(instance_file)
		end
	elseif edge_weight_type == "EUC_2D"
		readline(instance_file) # NODE_COORD_SECTION
		
		coordinates = zeros(Float64, n, 2)
		
		local line = readline(instance_file)
		while !contains(line, "DEMAND_SECTION")
			node, x, y = map(x -> parse(Int, x), split(line))
			coordinates[node, 1] = x
			coordinates[node, 2] = y
			
			line = readline(instance_file)
		end
		
		D = sqrt.((coordinates[:, 1] .- coordinates[:, 1]') .^ 2 + (coordinates[:, 2] .- coordinates[:, 2]') .^ 2 )
	else
		throw(ArgumentError("Invalid option $edge_weight_type"))
	end

	# Read demands
	demands = Vector{Int}(undef, n)
	local line = readline(instance_file)
	while !contains(line, "DEPOT_SECTION")
		node, demand = map(x -> parse(Int, x), split(line))
		demands[node] = demand
		
		line = readline(instance_file)
	end

	# Read depots
	depot = parse(Int, readline(instance_file))
	@assert contains(readline(instance_file), "-1")
	@assert depot == 1

	return CVRP(
		name = String(name),
		optimal_value = optimal_value,
		dimension = n,
		capacity = Q,
		C = D,
		q = demands
	)
end

# ╔═╡ 5201d169-3faf-4958-91b3-d27ffd3d879a
# Load all the instances
cvrp_instances = sort(
	[ 
		load_cvrp(instance_file) 
		for instance_file in readdir(data_dir, join=true) 
		if endswith(instance_file, "vrp")
	],
	by=(x -> x.dimension)
)

# ╔═╡ 48ca03a0-e18c-4505-8053-ab8a13478eae
"""
    solve_instances(instances::Vector{CVRP}, optimizer::Optimizer, formulate::Function) -> DataFrame

Solve a set of CVRP instances and return a DataFrame with results.

# Arguments
- `instances::Vector{CVRP}`: A vector of CVRP instances to be solved.
- `optimizer::Optimizer`: The optimizer to be used for solving the instances.
- `formulate::Function`: A function to formulate the CVRP instance into an optimization model.

# Returns
- `DataFrame`: A DataFrame containing the results for each instance, including instance name, size, 
  optimal values for ILP and LP, solution times, bounds, and gaps.

# Description
This function solves a list of CVRP instances using the provided optimizer for both ILP (Integer Linear Programming) 
and LP (Linear Programming) relaxations. For each instance, the function collects various metrics such as solution times, 
bounds, and optimal values, and compiles them into a DataFrame.

# Example
```julia
results = solve_instances(cvrp_instances, CPLEX.Optimizer, formulate_cvrp)
```

# Notes
- The function assumes the optimizer is installed and properly configured.
- `TIMEOUT` should be defined globally or replaced with a suitable value.
- The `formulate` function should correctly set up the optimization model for a CVRP instance.
"""       
function solve_instances(instances, optimizer, formulate)
	instance_names = String[]
	instance_sizes = Int[]
	
	ilp_opts = Float64[]
	upper_bounds = Float64[]
	lower_bounds = Float64[]
	gaps = Float64[]
	ilp_times = Float64[]

	lp_opts = Float64[]
	lp_times = Float64[]
	
	for instance in instances
		push!(instance_names, instance.name)
		push!(instance_sizes, instance.dimension - 1)
		
		model = Model(optimizer)
		set_time_limit_sec(model, TIMEOUT)
	
		_, (x, _) = formulate(instance, model)
	
		optimize!(model)

		push!(ilp_opts, instance.optimal_value)

		push!(upper_bounds, objective_value(model))
		push!(lower_bounds, objective_bound(model))
		push!(gaps, relative_gap(model))
		push!(ilp_times, solve_time(model))

		_ = relax_integrality(model)
		set_optimizer(model, optimizer)
		optimize!(model)
		
		push!(lp_opts, objective_value(model))
		push!(lp_times, solve_time(model))
	end

	df = DataFrame(
		"Instance" => instance_names,
		"Size" => instance_sizes,
		"LP OPT" => lp_opts,
		"ILP OPT" => ilp_opts,
		"LP time" => lp_times,
		"ILP time" => ilp_times,
		"Low. Bound" => lower_bounds,
		"Up. Bound" => upper_bounds,
		"Gap" => gaps
	)

	df
end

# ╔═╡ 04465162-a26f-4d9a-867a-4633351733c8
begin
	results_dir = "results"
 	if !isdir(results_dir)
	    mkdir(results_dir)
	end

	# Get all files in the directory
    files = readdir(results_dir, join=true)

    # Filter CSV files
    csv_files = filter(f -> endswith(f, ".csv"), files)

    if !isempty(csv_files)
		# Find the newest CSV file based on modification time
        newest_file = argmax(csv_files) do file
        	stat(file).mtime
    	end

		df = DataFrame(CSV.File(newest_file))
	else
		# Solve instances and save
		df = solve_instances(cvrp_instances, HiGHS.Optimizer, formulate)
		CSV.write(
			joinpath("$(results_dir)", "$(round(DateTime(now()), Dates.Minute)).csv"),
			df
		)
	end
		
	df
end

# ╔═╡ 2068fbcd-f1f4-4750-90d6-81f5e3833ce6
function formulate2(instance :: CVRP, model :: Model; relax=false)
	n = instance.dimension
	
	nodes = 1:n

	C = instance.C
	Q = instance.capacity
	q = instance.q

	# Customers
	cs = 2:n
	
	@variable(model, x[i=nodes, j=nodes; i != j], Bin)
	@variable(model, f[i=nodes, j=nodes, k=cs; i != j], Bin)

	# Kirchoff law
	@constraint(model, [j=cs], 
		1 == sum(x[j, :])
	)

	# Every node entered once
	@constraint(model, [j = cs],
		sum(x[:, j]) == 1
	)

	@constraint(model, [k=cs],
		sum(f[1, :, k]) == 1
	)
	@constraint(model, [k=cs],
		sum(f[:, k, k]) == 1
	)
	@constraint(model, [k=cs],
		sum(f[:, 1, k] ) == 0
	)
	@constraint(model, [k=cs],
		sum(f[k, :, k]) == 0
	)

	
	@constraint(model, [k=cs, i=cs; i != k],
		sum(f[i, :, k]) == sum(f[:, i, k])
	)
	
	
	@constraint(model, [i=nodes, j=nodes; i != j],
		sum(q[k] * f[i, j, k] for k in cs) <= Q * x[i, j]
	)

	@constraint(model, [k=cs, i=nodes, j=nodes; i != j], f[i, j, k] <= x[i, j])
	
	@objective(
        model,
        Min,
        sum(
			C[i, j] * x[i, j]
			for i ∈ nodes, j ∈ nodes
			if i != j
		)
    )

	model, (x, f)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CpuId = "adafc99b-e345-5852-983c-f28acb93d879"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Pipe = "b98c9c47-44ae-5843-9183-064241ee97a0"

[compat]
CSV = "~0.10.14"
CpuId = "~0.3.1"
DataFrames = "~1.6.1"
HiGHS = "~1.9.0"
JuMP = "~1.22.1"
Pipe = "~1.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.3"
manifest_format = "2.0"
project_hash = "22fff33894aa0012c9babbe0788ea0b15c280241"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "6c834533dc1fabd820c1db03c839bf97e45a3fab"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.14"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "9b1ca1aa6ce3f71b3d1840c538a8210a043625eb"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HiGHS]]
deps = ["HiGHS_jll", "MathOptInterface", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "a216e32299172b83abfe699604584f413ffbb045"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.9.0"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "9a550d55c49334beb538c5ad9504f07fc29a13dc"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.7.0+0"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "28f9313ba6603e0d2850fc3eae617e769c99bf83"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.22.1"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "9cc5acd6b76174da7503d1de3a6f8cf639b6e5cb"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.29.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "a3589efe0005fc4718775d8641b2de9060d23f73"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "363c4e82b66be7b9f7c7c7da7478fdae07de44b9"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.2"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "5d54d076465da49d6746c647022f3b3674e64156"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.8"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─0bb30050-dc8d-11ee-0b8c-8d66b5044644
# ╟─68564574-d1b2-470c-ab4e-0c221bc14895
# ╟─41796f80-e24e-4cbd-aa68-d1e608fa9a4f
# ╠═0edb3e3a-5f3d-4148-9c13-11b72b3ceef6
# ╠═08916fda-f183-4867-b62e-e9efc41fe4de
# ╠═7a487842-5fc4-4504-a7bb-4b3525f84e8a
# ╟─1ca827f2-9c53-448f-abb2-961c080c7b6f
# ╠═5201d169-3faf-4958-91b3-d27ffd3d879a
# ╟─5a38f991-19a2-47b3-9149-a6b650fcea76
# ╟─7c250684-5899-4cc7-a876-053130e13068
# ╟─8527a61c-5899-47f7-bf32-bc66dff5ae16
# ╟─4d052081-11dd-4780-835b-f8601ceb53ab
# ╠═dd8c3dd5-31e0-4549-8e4f-d1d5eaf11d6f
# ╟─3b3aed4f-cd75-41fd-bdb4-50bf1a7daefb
# ╟─2511a635-4115-43a0-8c66-96cf48990b08
# ╟─3eac7f08-6d6e-40aa-a939-97fe1d4e58e7
# ╟─732bf513-f9c5-4fc8-a89f-f1f99f26cb31
# ╟─fd7939f0-a2cf-4f16-abee-257031e67696
# ╟─7d199918-a41c-4434-b726-3cc770a67667
# ╟─3a43af06-e283-4fa4-928a-5c2d6c00b669
# ╠═bbf736fe-6d1d-4030-a625-7be6ddd03690
# ╟─0f47a17f-7d8f-47e8-ba93-505e86de8bcf
# ╠═04465162-a26f-4d9a-867a-4633351733c8
# ╟─2646ca7f-99e2-4bab-abde-d292b99a86f9
# ╠═26e41c3f-d6a1-4b71-a01c-09f0e6473dff
# ╟─0e27cbaa-6593-46e5-8d3d-1bd34174fc8a
# ╟─b72bde0e-00ee-4a51-8ab5-6463de37a260
# ╟─48ca03a0-e18c-4505-8053-ab8a13478eae
# ╟─2068fbcd-f1f4-4750-90d6-81f5e3833ce6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
