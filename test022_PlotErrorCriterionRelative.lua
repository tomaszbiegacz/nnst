require('init');
require('gnuplot');

--[[
    Plot Relative Error Criterion
--]]

function main()
    local criterion = nnst.ErrorCriterionRelative(0.8, 0.16)
    local e = 10
    local xs = torch.linspace(e-12, e+12, 200)

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
