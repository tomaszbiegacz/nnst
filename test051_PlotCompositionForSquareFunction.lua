require('init')
require('gnuplot')

--[[
    Plot composition of three perceptions to approximate square function    
--]]

function main()  

  local approximation = nnst.NetworkImporter:importFromJson([=[{
    "formatName":"nnst.NetworkScalarFunction",
    "formatVersion":1,
    "perceptonsInLayerCount":4,
    "transferFunction":"ReLU",
    "layers":[{
        "weight":[[-3.7],[-0.5],[0.5],[3.7]],
        "bias":[-5.1,-0.5,0.5,2.3]      
      },{
        "weight":[[-1,-1,-1,-1]]
      }]
  }]=])

  local x = torch.linspace(-2, 0, 20)  
  local y_approximation = approximation:forwardScalars(x)
  local y = -3*torch.pow(x+1, 2)

  gnuplot.plot({'predicted', x, y_approximation, '+-'}, {'expected', x, y, '+-'})
  gnuplot.grid(true)
end

main()
