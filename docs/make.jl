using Documenter
using PartitionnedStructures


using PartitionnedStructures.M_abstract_part_struct,  PartitionnedStructures.M_abstract_element_struct
using PartitionnedStructures.M_internal_elt_vec, PartitionnedStructures.M_internal_pv, PartitionnedStructures.M_elt_vec, PartitionnedStructures.M_elemental_elt_vec, PartitionnedStructures.M_elemental_pv
using PartitionnedStructures.M_elt_mat , PartitionnedStructures.M_part_mat , PartitionnedStructures.M_elemental_pm , PartitionnedStructures.M_elemental_em
using PartitionnedStructures.M_okoubi_koko , PartitionnedStructures.M_frontale 

makedocs(
  modules = [PartitionnedStructures],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "PartitionnedStructures.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(repo = "github.com/paraynaud/PartitionnedStructures.jl.git", devbranch = "main")
