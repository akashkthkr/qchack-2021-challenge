from typing import List, Tuple
import math

import numpy as np
import cirq

def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit],
    matrix: np.ndarray,
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """
    
    A method to convert a unitary matrix to a list of Sycamore operations.
    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.
    
    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
    """
    if target_qubits.count() == 1: # handled 1 qubit case.
        u_tilde = U_Tilde_Gate(matrix)
        if cirq.google.SYC_GATESET.is_supported_operation(u_tilde):
            return [cirq.Circuit(u_tilde(target_qubit[0]))], []
        return NotImplemented, []
    
    unitaries = decompose_to_two_level_unitaries(target_qubits.count(), matrix)
    ops = decompose_unitaries_to_ops(target_qubits, unitaries)
    
    if ops.count() == 0:
        return NotImplemented, []

    decomposed_ops []
    for op in ops:
        decomposed_op = cirq.optimizers.decompose_multi_controlled_rotation(cirq.unitary(op), op.controls, op.target)
        decomposed_ops.append(decomposed_op)
    
    # print(gates)
    return decomposed_op, []


def decompose_to_two_level_unitaries(
    target_qubits_count: int,
    matrix: np.ndarray
) -> List[Tuple[states: States, two_level_unitaries: np.ndarray]]:
    
    two_level_unitaries = []
    resultant_matrix = matrix #TODO:Put a check at the end to see if resultant is identity.
    n = 2**target_qubits_count
    for j in range(n - 2): # j is column index
        for i in range(j, n):
            unitary = np.identity(n)
            states = States(j, j, target_qubits_count)
            if resultant_matrix[i+1,j] != 0:
                if i = n: #Last iteration of rows
                    # Conjugating elements
                    for x in range(j + 1, n):
                        for y in range(j + 1, n):
                            unitary[x,y] = np.conjugate(resultant_matrix[x,y])
                else: 
                    normalizing_factor = np.sqrt(np.absolute(resultant_matrix[j,j])**2 + 
                    np.absolute(resultant_matrix[i+1,j])**2)

                    unitary[j,j] = np.conjugate(resultant_matrix[j,j])/normalizing_factor
                    unitary[j,i+1] = np.conjugate(resultant_matrix[i+1,j])/normalizing_factor
                    unitary[i+1,j] = resultant_matrix[i+1,j]/normalizing_factor
                    unitary[i+1,i+1] = -resultant_matrix[j,j]/normalizing_factor
                states = States(j, i+1, target_qubits_count)

            else:
                if i != j:
                    unitary[j,j] = np.conjugate(resultant_matrix[j,j])
            
            two_level_unitaries[states] = unitary
            resultant_matrix = np.matmul(unitary, resultant_matrix)
    return two_level_unitaries

def decompose_unitaries_to_ops(
    target_qubits: List[cirq.GridQubit],
    two_level_unitaries: List[Tuple[States, np.ndarray]],
) -> cirq.OP_TREE:
    
    gray_code = gray_code(target_qubits.count())
    total_ops = []
    for tuple in enumerate(two_level_unitaries):
        item_states = tuple[0]
        item_two_level_unitary = tuple[1]
        
        unitary_2x2 = get_2x2_unitary(item_two_level_unitary, item_states)
        u_tilde = U_Tilde_Gate(unitary_2x2)
        
        item_gray_codes = get_gray_code_set(gray_code, item_states)
        operations = []
        for i in range(item_gray_codes.count() - 2):
            target_qubit_index = get_changed_qubit_index(item_gray_codes[i], item_gray_codes[i + 1])
            if target_qubit_index == -1:
                return total_ops
                
            ops = get_operators((arget_qubit_index,item_gray_codes[i], target_qubits, cirq.X)
            operations.append(ops)
        
        u_tilde_operator = get_operators((arget_qubit_index,item_gray_codes[i], target_qubits, u_tilde)
        operations.append(u_tilde_operator)
        
        for i in range(operations.count() - 1, 0):
            operations.append(operations(i))
        for op in operations:
            total_ops.append()
    return total_ops

def get_operators(target_qubit_index: int, current_gray_code: str, target_qubits: List[cirq.GridQubit]
operator: cirq.SingleQubitGate):
    target_qubit = None
    zero_control_qubits = []
    one_control_qubits = []
    
    for i, qubit in target_qubits:
        if i == target_qubit_index:
            target_qubit = qubit
        if current_gray_code[i] == 0:
            zero_control_qubits.append(qubit)
        else:
            one_control_qubits.append(qubit)
    zero_c_gate = zero_bit_control_gate()
    ops = [cirq.Circuit(zero_bit_control_gate(zero_control_qubits, target_qubit)),
    cirq.Circuit(operator(target_qubit).controlled_by(one_control_qubits)) ]

    return ops

def zero_bit_control_gate(control_qubits, b, operator: cirq.SingleQubitGate):
    """Flips target bit when control bits are zero."""
    for control_qubit in control_qubits:
        yield cirq.X(control_qubit)
    yield cirq.Circuit(operator(b).controlled_by(control_bits))
    for control_qubit in control_qubits:
        yield cirq.X(control_qubit)
def keep_syc_gates(op):
    return isinstance(op, cirq.GateOperation) and cirq.google.SYC_GATESET.is_supported_operation(op.gate) 
def get_changed_qubit_index(current_state: str, next_state: str):
    for i in range(current_state.count()):
        if current_state[i] != next_state[i]:
            return i
    return -1
def get_2x2_unitary(two_level_unitary: np.ndarray, states: States):
    unitary_2x2 = np.ndarray
    unitary_2x2[1, 0] = two_level_unitary[states.get_row_index, states.get_row_index]
    unitary_2x2[0, 1] = two_level_unitary[states.get_row_index, states.get_column_index]
    unitary_2x2[1, 0] = two_level_unitary[states.get_column_index, states.get_row_index]
    unitary_2x2[1, 1] = two_level_unitary[states.get_column_index, states.get_column_index]
def get_gray_code_set(gray_codes: List[str], states: States) -> List[str]:
    start = gray_codes.index(states.get_start_state)
    end = gray_codes.index(states.get_end_state)
    gray_code_set = []
    for start in range(start, end):
        gray_code_set.append(gray_codes[start])
    return gray_code_set
def gray_code(n: str, ) -> List[str]:
    n = len(n)
    def gray_code_recurse (g: int, n: int) -> List[str]:
        k = len(g)
        if n <= 0:
            return
        else:
            for i in range (k-1, -1, -1):
                char='1'+g[i]
                g.append(char)
            for i in range (k-1, -1, -1):
                g[i]='0'+g[i]
            gray_code_recurse (g, n-1)
    g = ['0', '1']
    gray_code_recurse(g, n-1)
    return g
class States:
    def __init__(self, start_state: int, end_state: int, total_target_qubits: int):
        """ Constructor """
        self.__start_state = np.binary_repr(start_state, width=total_target_qubits)
        self.__end_state = np.binary_repr(end_state, width=total_target_qubits)
        self.__i = start_state
        self.__j = end_state
    
    def get_start_state(self):
        return self.__start_state
    
    def get_end_state(self):
        return self.__end_state
    
    def get_row_index(self):
        return self.__i

    def get_column_index(self):
        return self.__j

class U_Tilde_Gate(cirq.SingleQubitGate):
    def __init__(self, unitary: np.ndarray):
        self.__unitary = unitary

    def _unitary_(self):
        return np.array([[self.__unitary[1, 0],self.__unitary[0, 1]], 
        [self.__unitary[1, 0], self.__unitary[1, 1]]])
    
    def __str__(self):
        return 'U~' 