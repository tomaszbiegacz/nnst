require('init');

--[[
    Learning linear function via percepton 
    with training set having ordered values.
--]]

function learn(net, errCriterion, x, y, exporter)
  net:zeroGradParameters()

  local xInput = torch.Tensor{ x }
  local yActual = net:forward(xInput)
  local yExpected = torch.Tensor{ y }
  local error = errCriterion:forward(yActual, yExpected)  

  local de, dde = errCriterion:backward2(yActual, yExpected)
  net:backward2(xInput, de, dde)  

  exporter:addState(x, y, error, de[1], dde[1], net)
  net:updateParametersNewton()
end

function main()
  local net = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,
    "layers":[{
        "weight":[[5]],
        "bias":[5]
      }]
  }]=])

  local errCriterion = nnst.ErrorCriterionSquareRelative()

  local maxSampleValue = 100
  local x = torch.linspace(1, maxSampleValue, maxSampleValue)
  local y = 2*x+1

  local exporter = nnst.ExportPercepton()

  local lessonsCount = 100
  for i=1,lessonsCount do
    learn(net, errCriterion, x[i], y[i], exporter)
  end

  exporter:exportToCsv('testResults/test040_LearnPerceptonProgresive.csv')
end

main()
