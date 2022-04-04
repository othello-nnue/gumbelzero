using Bits
toplane(a::UInt64) = reshape(bits(a), 8, 8, 1, 1)

function bitboard_to_array(a::UInt64)
    ret = []
    sizehint!(ret, count_ones(a))
    while a != 0
        push!(ret, UInt8(scan1(a) - 1))
        a &= a - 1
    end
    return ret
end