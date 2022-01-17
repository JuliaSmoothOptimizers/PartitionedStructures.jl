using Documenter
using PartitionedStructures


using PartitionedStructures.M_abstract_part_struct,  PartitionedStructures.M_abstract_element_struct, PartitionedStructures.M_part_mat,  PartitionedStructures.M_part_v
using PartitionedStructures.M_internal_elt_vec, PartitionedStructures.M_internal_pv, PartitionedStructures.M_elt_vec, PartitionedStructures.M_elemental_elt_vec, PartitionedStructures.M_elemental_pv
using PartitionedStructures.M_elt_mat, PartitionedStructures.M_elemental_pm , PartitionedStructures.M_elemental_em
using PartitionedStructures.M_okoubi_koko , PartitionedStructures.M_frontale, PartitionedStructures.M_1_parallel, PartitionedStructures.M_2_parallel, PartitionedStructures.M_3_parallel
using PartitionedStructures.M_utils

makedocs(
  # modules = [PartitionedStructures],
	modules = [PartitionedStructures, M_abstract_part_struct,M_abstract_element_struct,M_internal_elt_vec,M_internal_pv,M_elt_vec,M_elemental_elt_vec,M_elemental_pv,M_elt_mat,M_part_mat,M_elemental_pm,M_elemental_em,M_okoubi_koko,M_frontale, M_1_parallel, M_2_parallel, M_3_parallel, M_utils],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "PartitionedStructures.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(repo = "github.com/paraynaud/PartitionedStructures.jl.git", devbranch = "main")
