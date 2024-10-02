function yEstimate = CalculateEstimation(chromosome, x, nrOfVariableRegisters, constantRegister)

    nrOfGenes = length(chromosome);
    operands = [x zeros(1,nrOfVariableRegisters-1) constantRegister];
    
    for i = 1:4:(nrOfGenes-3)
        
        operator = chromosome(i);
        destination = chromosome(i+1);
        operand1 = operands(chromosome(i+2));
        operand2 = operands(chromosome(i+3));
    
        if operator == 1
                operands(destination) = operand1 + operand2;

        elseif operator == 2
                operands(destination) = operand1 - operand2;

        elseif operator == 3
                operands(destination) = operand1 * operand2;

        else
            if operand2 == 0 
                operands(destination) = 10^12;
            else
                operands(destination) = operand1 / operand2;
            end
        end
    end

    yEstimate = operands(1);
end