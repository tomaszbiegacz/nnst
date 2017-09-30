require('init');
require('gnuplot');

--[[
    Plot Peception Error with ALU and error Square Relative
--]]

function main()
    local net = nnst.NetworkImporter:importFromJson([=[{
        "formatName":"nnst.NetworkScalarFunction",
        "formatVersion":1,
        "layers":[{
            "weight":[[5]],
            "bias":[5]
        }]
    }]=])

    local layer = net:getInputLayer()
    local errCriterion = nnst.ErrorCriterionSquareRelative()
    
    local yExpected = torch.Tensor{ 5 }

    local count = 100
    local xInput = torch.Tensor{ 2 }
    local aValues = torch.linspace(-5, 3, count)

    local e = aValues:clone()
    e:apply(function (a)
        net:zeroGradParameters()        
        layer:setWeight(1, 1, a)

        local yActual = net:forward(xInput)
        local error = errCriterion:forward(yActual, yExpected)

        return error
    end)

    local de = aValues:clone()
    de:apply(function (a)
        net:zeroGradParameters()        
        layer:setWeight(1, 1, a)

        local yActual = net:forward(xInput)
        local error = errCriterion:forward(yActual, yExpected)

        local de = errCriterion:backward(yActual, yExpected)
        net:backward(xInput, de)  

        return layer:getWeightGrad(1, 1)
    end)

    local dde = aValues:clone()
    dde:apply(function (a)
        net:zeroGradParameters()        
        layer:setWeight(1, 1, a)

        local yActual = net:forward(xInput)
        local error = errCriterion:forward(yActual, yExpected)

        local de, dde = errCriterion:backward2(yActual, yExpected)
        net:backward2(xInput, de, dde)  

        return layer:getWeightGrad2(1, 1)
    end)

    gnuplot.plot({'e', aValues, e, '-'}, {'de', aValues, de, '-'}, {'dde', aValues, dde, '-'})
    gnuplot.grid(true)
end

main()
