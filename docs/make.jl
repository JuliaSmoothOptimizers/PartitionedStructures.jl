using Documenter
using PartitionedStructures

using PartitionedStructures.M_abstract_part_struct,
  PartitionedStructures.M_abstract_element_struct,
  PartitionedStructures.M_part_mat,
  PartitionedStructures.M_part_v
using PartitionedStructures.M_elt_vec,
  PartitionedStructures.ModElemental_ev,
  PartitionedStructures.ModElemental_pv
using PartitionedStructures.M_elt_mat,
  PartitionedStructures.ModElemental_pm, PartitionedStructures.ModElemental_em
using PartitionedStructures.ModElemental_elo_bfgs,
  PartitionedStructures.ModElemental_plo_bfgs,
  PartitionedStructures.ModElemental_elo_sr1,
  PartitionedStructures.ModElemental_plo,
  PartitionedStructures.ModElemental_plo_sr1
using PartitionedStructures.Utils,
  PartitionedStructures.Link,
  PartitionedStructures.Instances,
  PartitionedStructures.PartitionedQuasiNewton,
  PartitionedStructures.PartitionedLOQuasiNewton
using PartitionedStructures.PartMatInterface

makedocs(
  modules = [
    PartitionedStructures,
    M_abstract_part_struct,
    M_abstract_element_struct,
    M_internal_elt_vec,
    M_internal_pv,
    M_elt_vec,
    ModElemental_ev,
    ModElemental_pv,
    M_elt_mat,
    M_part_mat,
    ModElemental_pm,
    ModElemental_em,
    M_okoubi_koko,
    M_frontale,
    M_1_parallel,
    M_2_parallel,
    M_3_parallel,
    Link,
    Utils,
    PartitionedQuasiNewton,
    PartitionedLOQuasiNewton,
    ModElemental_elo_bfgs,
    ModElemental_plo_bfgs,
    ModElemental_plo_sr1,
    ModElemental_elo_sr1,
    ModElemental_plo,
    Instances,
    PartMatInterface,
  ],
  doctest = true,
  linkcheck = true,
  strict = false,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "PartitionedStructures.jl",
  pages = Any[
    "Home" => "index.md",
    "Tutorial" => "tutorial.md",
    "Reference" => "reference.md",
    "Developer note" => "developer_note.md",
  ],
)

deploydocs(repo = "github.com/paraynaud/PartitionedStructures.jl.git", devbranch = "main")
