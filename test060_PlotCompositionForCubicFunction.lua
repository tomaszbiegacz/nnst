require('init')
require('gnuplot')

--[[
    Plot composition of three perceptions to approximate cubic function
    Inspired by <https://medium.com/towards-data-science/can-neural-networks-really-learn-any-function-65e106617fc6>.
--]]

function main()

  local n1 = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,  
    "layers":[{
        "weight":[[-5]],
        "bias":[-7.7]      
      }]    
  }]=])

  local n2 = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,  
    "layers":[{
        "weight":[[-1.2]],
        "bias":[-1.3]      
      }]    
  }]=])

  local n3 = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,  
    "layers":[{
        "weight":[[1.2]],
        "bias":[1]      
      }]    
  }]=])

  local approximation = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,
    "perceptonsInLayerCount":3,
    "transferFunction":"ReLU",
    "layers":[{
        "weight":[[-5],[-1.2],[1.2]],
        "bias":[-7.7,-1.3,1]      
      },{
        "weight":[[-1,-1,-1]]
      }]
  }]=])

  local x = torch.linspace(-2, 1.5, 20)
  local y_n1 = n1:forwardScalars(x)
  local y_n2 = n2:forwardScalars(x)
  local y_n3 = n3:forwardScalars(x)
  local y_approximation = approximation:forwardScalars(x)
  local y = torch.pow(x, 3) + torch.pow(x, 2) - x - torch.ones(x:size())

  gnuplot.plot({'n1', x, y_n1, '+-'}, {'n2', x, y_n2, '+-'}, {'n3', x, y_n3, '+-'}, {'f(x)', x, y_approximation, '+-'}, {'y', x, y, '+-'})
  gnuplot.grid(true)
end

main()
