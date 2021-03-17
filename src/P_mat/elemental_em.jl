module	M_elemental_em

	mutable struct Elemental_ev{T}
		nie :: Int # nᵢᴱ
		indices :: Vector{Int} # size nᵢᴱ
		hie :: Matrix{T} # size nᵢᴱ × nᵢᴱ
	end

end 