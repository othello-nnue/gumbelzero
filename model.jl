using Flux

act = mish
#use testmode! for testing

#pyramid-net-like-block
block(nf::Integer) = SkipConnection(Chain(
    BatchNorm(nf),
    Conv((3, 3), nf => nf, pad=SamePad()),
    BatchNorm(nf, act),
    Conv((3, 3), nf => nf, pad=SamePad()),
    BatchNorm(nf),
), +)


#resnext-like-block
resnextblock(nf::Integer) = SkipConnection(Chain(
    BatchNorm(nf),
    DepthwiseConv((3, 3), nf => nf, pad=SamePad()),
    BatchNorm(nf),
    Conv((1, 1), nf => 4 * nf, pad=SamePad()),
    BatchNorm(4 * nf, act),
    Conv((1, 1), 4 * nf => nf, pad=SamePad()),
    BatchNorm(nf)
), +)

function widening(nf::Integer, input)
    s = size(input)
    s[3] = nf - s[3]
    pad = zeros(s)
    return cat(input, pad; dims=3)
end

narrowing(nf::Integer, input) = input[:, :, 1:2, :]

nf = 32 #number of filters
# model = Chain(
#     Conv((3, 3), 2=>nf, act, pad=SamePad()),
#     Conv((3, 3), nf=>nf, act, pad=SamePad()),
#     Conv((3, 3), nf=>nf, act, pad=SamePad()),
#     Conv((3, 3), nf=>nf, act, pad=SamePad()),
#     Conv((3, 3), nf=>nf, act, pad=SamePad()),
#     Conv((3, 3), nf=>nf, act, pad=SamePad()),
#     Conv((3, 3), nf=>1, tanh, pad=SamePad()),
# )
#bigmodel = Chain(block(nf), block(nf), block(nf), block(nf), block(nf), block(nf))

RADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) =
  Flux.Optimiser(RADAM(1, β), WeightDecay(decay), Descent(η))
