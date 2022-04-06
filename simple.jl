#simple expected result learning
#gumbel without policy

include("othello.jl");
include("model.jl");
using Bits
using Flux
using CUDA

nf = 32
model = Chain(
    #Conv((1, 1), 2 => nf, mish, pad=SamePad()),
    block(nf), block(nf), block(nf), block(nf),
    Conv((1, 1), nf => 1, tanh, pad=SamePad()),
)

toplane(a::UInt64) = reshape(bits(a), 8, 8, 1, 1)
input(a::Game) = cat(toplane(a.a), toplane(a.b), zeros(Float32, 8, 8, nf - 2), dims=3)
output(a::Game) = model(input(a))
value(a::Game) = sum(output(a))
value_target(a::UInt64) = toplane(a) .* 2 .- 1
#input(a::Vector{Game}) = cat((input).(a)..., dims = 4)


#higher is better
function against_random()
    turn = rand(Bool)

    g = init()
    while notend(g)
        if rawmoves(g) == 0
            g = flip(g)
        else
            turn = !turn
        end
        if turn
            move = agent(value, g)
        else
            move = rand_agent(g)
        end
        g = g + move
    end
    score = count_ones(g.a)
    if turn
        score = 64 - score
    end
    return score
end

function generate_traindata!(positions, values)
    turns = []
    turn = false
    g = init()
    while notend(g)
        if rawmoves(g) == 0
            g = flip(g)
        else
            turn = !turn
        end

        push!(positions, g)
        push!(turns, turn)

        pair = (x -> (x, -value(g + x))).(moves(g))
        maxmove = findmax((x -> sum(x[2])).(pair))[2]

        if rand() < epsilon
            move = rand(pair)[1]
        else
            move = pair[maxmove][1]
        end

        g = g + move
    end
    value_data = ((x == turn ? g.b : ~g.b) for x in turns)
    append!(values, value_data)
end

opt = RADAMW(0.1, (0.9, 0.999), 1)
epsilon = 0.2

for i in 1:10000000000000

    testmode!(model)

    x_train::Vector{Game} = []
    y_train::Vector{UInt64} = []

    while length(x_train) <= 256
        generate_traindata!(x_train, y_train)
    end

    #model |> gpu
    global opt

    testmode!(model, false)

    # augment
    x_train = vcat((augment).(x_train)...)
    y_train = vcat((Othello.augment).(y_train)...)

    #transform
    x_train2 = cat((input).(x_train)..., dims=4)
    y_train2 = cat((value_target).(y_train)..., dims=4)

    parameters = params(model)

    data = (x_train2, y_train2)
    loader = Flux.Data.DataLoader(data, batchsize=64, shuffle=true)

    loss(x, y) = Flux.Losses.mse(model(x), y)

    for epoch in 1:10
        Flux.train!(loss, parameters, loader, opt)
    end

    if i % 5 == 0
        t = (x -> against_random()).(1:10)
        print(sum(t) / length(t), "\n")
    end

    #model |> cpu
end
