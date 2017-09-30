local ErrorCriterionSquare, parent = torch.class('nnst.ErrorCriterionSquare', 'nnst.ErrorCriterion')

--[[
    Square error criterion continous up to second derivative
--]]

function ErrorCriterionSquare:__init()
    parent.__init(self)
end

function ErrorCriterionSquare:updateOutput(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local x = input[1]
    local e = target[1]

    local diff = x - e
    return diff*diff
end

function ErrorCriterionSquare:updateGradInput(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local x = input[1]
    local e = target[1]

    local diff = x - e
    return torch.Tensor{ 2*diff }
end

function ErrorCriterionSquare:updateGrad2Input(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")
    
    return torch.Tensor{ 2 }
end

function ErrorCriterionSquare:__tostring__()
  return string.format('%s', torch.type(self))
end
