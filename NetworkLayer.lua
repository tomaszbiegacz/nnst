local NetworkLayer, _ = torch.class('nnst.NetworkLayer')

function NetworkLayer:__init(linearLayer, transferLayer)
    assert(linearLayer, "linearLayer")

    self.linear = linearLayer
    self.transfer = transferLayer
end

-- linear

function NetworkLayer:getLinearOutput(perceptonPos)
    return self.linear.output[perceptonPos]
end

-- weight

function NetworkLayer:getWeight(perceptonPos, inputPos)
    local inputs = self.linear.weight[perceptonPos]
    return inputs[inputPos]
end

function NetworkLayer:getWeightGrad(perceptonPos, inputPos)
    local inputs = self.linear.gradWeight[perceptonPos]
    return inputs[inputPos]
end

function NetworkLayer:getWeightGrad2(perceptonPos, inputPos)
    local inputs = self.linear.grad2Weight[perceptonPos]
    return inputs[inputPos]    
end

function NetworkLayer:setWeight(perceptonPos, inputPos, value)
    local inputs = self.linear.weight[perceptonPos]
    inputs[inputPos] = value
end

-- bias

function NetworkLayer:getBias(perceptonPos)
    assert(self.linear.bias, "self.linear.bias");
    return self.linear.bias[perceptonPos]
end

function NetworkLayer:getBiasGrad(perceptonPos)
    assert(self.linear.bias, "self.linear.bias");
    return self.linear.gradBias[perceptonPos]
end

function NetworkLayer:getBiasGrad2(perceptonPos)
    assert(self.linear.bias, "self.linear.bias");
    return self.linear.grad2Bias[perceptonPos]
end

function NetworkLayer:setBias(perceptonPos, value)
    assert(self.linear.bias, "self.linear.bias");
    self.linear.bias[perceptonPos] = value
end

-- transfer

function NetworkLayer:hasTransfer()
    if self.transfer then
        return true
    else
        return false
    end
end

function NetworkLayer:getTransferOutput(perceptonPos)
    assert(self.transfer, self.transfer);
    return self.transfer.output[perceptonPos];
end

function NetworkLayer:getTransferGradInput(perceptonPos)
    assert(self.transfer, self.transfer);
    return self.transfer.gradInput[perceptonPos];
end

function NetworkLayer:getTransferGradForInput(perceptonPos)
    assert(self.transfer, self.transfer);
    return self.transfer.gradForInput[perceptonPos];
end

function NetworkLayer:getTransferGrad2Input(perceptonPos)
    assert(self.transfer, self.transfer);
    return self.transfer.grad2Input[perceptonPos];
end

function NetworkLayer:getTransferGrad2ForInput(perceptonPos)
    assert(self.transfer, self.transfer);
    return self.transfer.grad2ForInput[perceptonPos];
end

-- support

function NetworkLayer:__tostring__()    
    return string.format('%s, linear:%s transfer:%s', torch.type(self), self.linear, self.transfer)     
end

function NetworkLayer:getState(args)
    local resultLayer = {}

    resultLayer.weight = torch.totable(self.linear.weight)
    if args.includeGrad then
        resultLayer.gradInput = torch.totable(self.linear.gradInput)
        if self.linear.grad2Input then
          resultLayer.grad2Input = torch.totable(self.linear.grad2Input)
        end

        resultLayer.gradWeight = torch.totable(self.linear.gradWeight)
        if self.linear.grad2Weight then
          resultLayer.grad2Weight = torch.totable(self.linear.grad2Weight)
        end
    end

    if self.linear.bias then
        resultLayer.bias = torch.totable(self.linear.bias)
        if args.includeGrad then
            resultLayer.gradBias = torch.totable(self.linear.gradBias)
            if self.linear.grad2Bias then
              resultLayer.grad2Bias = torch.totable(self.linear.grad2Bias)
            end
        end
    end

    if args.includeOutput then
        resultLayer.outLinear = torch.totable(self.linear.output)
    end

    if self.transfer then
        if args.includeGrad then
            resultLayer.gradTransfer = torch.totable(self.transfer.gradInput)
            if self.transfer.grad2Input then
              resultLayer.grad2Transfer = torch.totable(self.transfer.grad2Input)
            end
        end

        if args.includeOutput then
            resultLayer.outTransfer = torch.totable(self.transfer.output)
        end
    end

    return resultLayer
end