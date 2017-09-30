local inspect = require('inspect');
local NetworkSequential, parent = torch.class('nnst.NetworkSequential', 'nn.Sequential')

--
-- NetworkSequential
--

function NetworkSequential:__init()
    parent.__init(self)

    self.layers = {}        
end

function NetworkSequential:_addPerceptonLayer(linearLayer, transferLayer)
    local networkLayer = nnst.NetworkLayer(linearLayer, transferLayer)    
    self:add( networkLayer.linear )

    if networkLayer:hasTransfer() then        
        self:add( networkLayer.transfer )
    end

    table.insert(self.layers, networkLayer)
end

function NetworkSequential:getState(args)
    local stateArgs = args or {}
    local result = {}

    for i=1,table.getn(self.layers) do
        local networkLayer = self.layers[i]        
        table.insert(result, networkLayer:getState(stateArgs))
    end

    return result
end

function NetworkSequential:getStateDescription()    
    return inspect(self:getState{ 
        includeGrad = true,
        includeOutput = true
    })
end

function NetworkSequential:getInputLayer()
    assert(table.getn(self.layers) >= 1, "table.getn(self.layers) >= 1")    

    return self.layers[1]
end

function NetworkSequential:getHiddenLayer(layerPos)
    assert(table.getn(self.layers) >= 3, "table.getn(self.layers) >= 3")
    assert(layerPos >= 1, "layerPos >= 1")
    assert(layerPos + 1 < table.getn(self.layers), "layerPos + 1 < table.getn(self.layers)")    

    return self.layers[layerPos + 1]
end

function NetworkSequential:getOutputLayer()
    assert(table.getn(self.layers) >= 1, "table.getn(self.layers) >= 1")

    return self.layers[table.getn(self.layers)]
end

function NetworkSequential:backward2(input, gradOutput, grad2Output, scale)
   scale = scale or 1
   local currentGradOutput = gradOutput
   local currentGrad2Output = grad2Output
   local currentModule = self.modules[#self.modules]

   for i=#self.modules-1,1,-1 do     
      local previousModule = self.modules[i]
      currentGradOutput, currentGrad2Output = currentModule:backward2(previousModule.output, currentGradOutput, currentGrad2Output, scale)
      
      currentModule = previousModule
   end
   currentGradOutput, currentGrad2Output = currentModule:backward2(input, currentGradOutput, currentGrad2Output, scale)   
   
   return currentGradOutput, currentGrad2Output
end

function NetworkSequential:updateParametersNewton(learningRate)
    learningRate = learningRate or 1
    self:applyToModules(function(module) module:updateParametersNewton(learningRate) end)
end