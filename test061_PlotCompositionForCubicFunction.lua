require('init')
require('gnuplot')

--[[
    Plot composition of three perceptions to approximate cubic function
    Inspired by <https://medium.com/towards-data-science/can-neural-networks-really-learn-any-function-65e106617fc6>.
--]]

function main()  

  local approximation = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,
    "perceptonsInLayerCount":3,
    "transferFunction":"ReLU",
    "layers":[{
        "weight":[[-4.6],[-1.2],[1.2]],
        "bias":[-7.2,-1.3,1]      
      },{
        "weight":[[-1,-1,-1]]
      }]
  }]=])

  local x = torch.linspace(-2, 0, 20)  
  local y_approximation = approximation:forwardScalars(x)
  local y = torch.pow(x, 3) + torch.pow(x, 2) - x - torch.ones(x:size())

  gnuplot.plot({'predicted', x, y_approximation, '+-'}, {'expected', x, y, '+-'})
  gnuplot.grid(true)
end

main()
