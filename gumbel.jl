# Simple Gumbel Alphazero with n = 2
# Don't even need to implement tree search

using Random

"""
...
# Arguments
- `logits`: policy logits
- `topk`: the number of actions to select. 
...
returns gumbel added logits, indexed by indicies 
and indices
"""

function select(logits, topk)
    gumbel = -log(Random.randexp(size(logits))) #for later use
    gumbeled = logits + gumbel
    indices = partialsortperm(gumbeled, topk, rev=true)
    return gumbeled[indices], indices
end

function move(gumbel_added_logits, indices, values)
    #t = findmax(map(x -> values[x[1]] + gumbel_added_logits[x[2]], enumerate(indices)))
    #should be `view(::Vector{Int64}, 1:3)`
    t = findmax(values + gumbel_added_logits)
    return indices[t[2]]
end

#give child_values=0 when unvisited
function completed_value(value, policy, visits, child_values)
    policy_sum = sum(policy .* (visits .> 0))
    value_sum = value + sum(visits)/policy_sum * sum(policy .* child_value)
    total_visits = 1 + sum(visits)
    return value_sum/total_visits
end


#loss : kl(policy_target, network)
function policy_target_logits(logits, values)
    return logits + values
end
