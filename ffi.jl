module Othello
using Libdl
# currently compiled for skylake
# to use on colab
#othello = Libdl.dlopen("./othello.dll")
othello = Libdl.dlopen("./libothello.so")
sym_moves = Libdl.dlsym(othello, :moves)
sym_flip = Libdl.dlsym(othello, :flip)
sym_stable = Libdl.dlsym(othello, :stable)

export moves, flip, augment

moves(a::UInt64, b::UInt64)::UInt64 = ccall(sym_moves, UInt64, (UInt64, UInt64), a, b)
flip(a::UInt64, b::UInt64, c::UInt8)::UInt64 = ccall(sym_flip, UInt64, (UInt64, UInt64, UInt8), a, b, c)
stable(a::UInt64, b::UInt64)::UInt64 = ccall(sym_stable, UInt64, (UInt64, UInt64), a, b)

ops = [Libdl.dlsym(othello, :vert), Libdl.dlsym(othello, :hori), Libdl.dlsym(othello, :diag), Libdl.dlsym(othello, :gaid)]
t(i, a) = ccall(ops[i], UInt64, (UInt64,), a)
half(a::UInt64)::Vector{UInt64} = [a, t(2, a), t(3, a), t(4, a)]
augment(a::UInt64)::Vector{UInt64} = vcat(half(a), half(t(1, a)))

using Test
@test 0x0000_1020_0408_0000 == moves(0x0000_0008_1000_0000, 0x0000_0010_0800_0000)
@test (UInt64(1) .<< [10, 13, 17, 46, 50, 53, 22, 41]) == augment(UInt64(1) << 10)
end