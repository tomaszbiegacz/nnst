local ErrorCriterionSquareRelative, parent = torch.class('nnst.ErrorCriterionSquareRelative', 'nnst.ErrorCriterion')

--[[
    Square of relative error criterion continous up to second derivative
--]]

function ErrorCriterionSquareRelative:__init()
    parent.__init(self)
end

function ErrorCriterionSquareRelative:updateOutput(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local x = input[1]
    local e = target[1]

    local diff = (x-e)/e
    return diff*diff
end

function ErrorCriterionSquareRelative:updateGradInput(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local x = input[1]
    local e = target[1]

    local diff = 2*(x-e)/(e*e)
    return torch.Tensor{ diff }
end

function ErrorCriterionSquareRelative:updateGrad2Input(input, target)
    assert(input:nDimension() == 1 and input:size(1) == 1, "input:nDimension() == 1 and input:size(1) == 1")
    assert(target:nDimension() == 1 and target:size(1) == 1, "target:nDimension() == 1 and target:size(1) == 1")

    local e = target[1]
    
    return torch.Tensor{ 2/(e*e) }
end

function ErrorCriterionSquareRelative:__tostring__()
  return string.format('%s', torch.type(self))
end
