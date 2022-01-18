module Utils

	using LinearAlgebra
	my_and = (a :: Bool,b :: Bool) -> (a && b)


	"""
			BFGS(s, y, B)
	Perform the BFGS update over the matrix B by using the vector s and y.
	"""
	BFGS(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); BFGS!(s,y,B,B_1;kwargs...); B_1 end
	BFGS(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); BFGS!(x_1 - x, g_1 - g, B, B_1; kwargs...); B_1 end 
	BFGS!(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}; kwargs...) where Y <: Number = BFGS!(x_1 - x, g_1 - g, B, B_1; kwargs...)
	BFGS!(s :: Vector{Y}, y :: Vector{Y}, B :: Symmetric{Y,Matrix{Y}}, B_1 :: Symmetric{Y,Matrix{Y}}; kwargs...) where Y <: Number = BFGS!(s,y,B.data, B_1.data; kwargs...)
	function BFGS!(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}; index=0, reset=4) where Y <: Number #Array that will store the next approximation of the Hessian
		if dot(s,y) > 0  # curvature condition
			Bs = B * s 
			terme1 =  y * y' / dot(y,s)
			terme2 = Bs * Bs' / dot(Bs,s)
			B_1 .= B .+ terme1 .- terme2
		elseif index < reset #
			B_1 .= B 
		else
			n = length(s)
			B_1 .= reshape([ (i==j ? (Y)(1) : (Y)(0)) for i = 1:n for j =1:n], n, n)
		end 
	end

	"""
			SR1(s, y, B)
	Perform the BFGS update over the matrix B by using the vector s and y.
	"""
	SR1(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); SR1!(s,y,B,B_1;kwargs...); B_1 end
	SR1(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}; kwargs...) where Y <: Number = begin B_1=similar(B); SR1!(x_1 - x, g_1 - g, B, B_1; kwargs...); B_1 end 
	SR1!(x :: Vector{Y}, x_1 :: Vector{Y}, g :: Vector{Y}, g_1 :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}) where Y <: Number = SR1!(x_1 - x, g_1 - g, B, B_1)	
	SR1!(s :: Vector{Y}, y :: Vector{Y}, B :: Symmetric{Y,Matrix{Y}}, B_1 :: Symmetric{Y,Matrix{Y}}; kwargs...) where Y <: Number = SR1!(s,y,B.data, B_1.data; kwargs...)
	function SR1!(s :: Vector{Y}, y :: Vector{Y}, B :: Array{Y,2}, B_1 :: Array{Y,2}; index=0, reset=4, ω = 1e-6) where Y <: Number
		r = y .- B*s
		if abs(dot(s,r)) > ω * norm(s,2) * norm(r,2)
			B_1 .= B .+ ((r * r')./dot(s,r))
		elseif index < reset #
			B_1 .= B 	
		else
			n = length(s)
			B_1 .= reshape([ (i==j ? (Y)(1) : (Y)(0)) for i = 1:n for j =1:n], n, n)
		end 
	end
			
	
	

 	export BFGS, BFGS!, SR1, SR1!
	export my_and
end
