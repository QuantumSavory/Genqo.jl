from juliacall import Main as jl

jl.seval("import Pkg")
jl.Pkg.activate(".")
jl.seval("using Genqo")

from .genqo import GenqoBase, TMSV, SPDC, ZALM, SIGSAG, k_function_matrix
