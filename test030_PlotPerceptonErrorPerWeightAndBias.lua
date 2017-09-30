require('nn');
require('gnuplot');

--[[
    Plot percepton error as a function of it's weight and bias
--]]

local a = torch.linspace(-10, 10, 100)
local b = torch.linspace(-10, 10, 100)

local s_a = torch.Tensor(a:size(1),b:size(1)):zero()
for i_b=1,b:size(1) do
    s_a[{{}, i_b}] = a
end

local s_b = torch.Tensor(a:size(1),b:size(1)):zero()
for i_a=1,a:size(1) do
    s_b[{i_a, {}}] = b
end

local x = 2
local y = torch.Tensor{2}

-- see https://github.com/torch/nn/blob/master/doc/criterion.md#abscriterion
local f_e = nn.AbsCriterion();

-- see https://github.com/torch/nn/blob/master/doc/transfer.md#elu
local f_t = nn.ELU(1);

local s_f = torch.Tensor(a:size(1),b:size(1)):zero()
for i_a=1,a:size(1) do
    for i_b=1,b:size(1) do
        local net = x*s_a[{i_a, i_b}] + s_b[{i_a, i_b}]
        local output = f_t:forward(torch.Tensor{net})
        s_f[{i_a, i_b}] = f_e:forward(output, y);
    end
end

gnuplot.setterm('wxt')
gnuplot.splot(s_a, s_b, s_f);
