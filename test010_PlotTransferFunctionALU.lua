require('init');
require('gnuplot');

--[[
    Plot ALU transfer function
--]]

function main()
    local m = nnst.TransferFunctionALU()
    local xs = torch.linspace(-7, 1, 200)
    local go = torch.ones(xs:size(1))

    local f = m:forward(xs)
    m:backward2(xs, go, go)

    local df = m.gradForInput
    local ddf = m.grad2ForInput
    
    gnuplot.plot({'f(x)', xs, f, '-'}, {'df', xs, df, '-'}, {'d^2f', xs, ddf, '-'})    
    
    gnuplot.grid(true)
end

main()
