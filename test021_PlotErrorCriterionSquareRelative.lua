require('init');
require('gnuplot');

--[[
    Plot Square Relative Error Criterion
--]]

function main()
    local criterion = nnst.ErrorCriterionSquareRelative()
    local e = 5
    local xs = torch.linspace(-2, 12, 100)

    local f = xs:clone()
    f:apply(function (x)
        return criterion:forward(torch.Tensor{ x }, torch.Tensor{ e })
    end)

    local df = xs:clone()
    df:apply(function (x)
        local result = criterion:backward(torch.Tensor{ x }, torch.Tensor{ e })
        return result[1]
    end)

    local ddf = xs:clone()
    ddf:apply(function (x)
        local _, result = criterion:backward2(torch.Tensor{ x }, torch.Tensor{ e })
        return result[1]
    end)

    gnuplot.plot({'f(x)', xs, f, '-'}, {'df', xs, df, '-'}, {'d^2f', xs, ddf, '-'})
    gnuplot.grid(true)
end

main()
