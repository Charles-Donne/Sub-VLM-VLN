"""
Prompt templates for LLM-based navigation planning
All prompts are in English for better LLM performance
"""

# Initial planning prompt - generates first subtask at task start
INITIAL_PLANNING_PROMPT = """You are the planning and thinking module of a Vision-Language Navigation agent. Your role is to analyze the spatial environment and design an easily achievable subtask for the low-level action execution module.

# Overall Navigation Task Instruction (Global Task)
{instruction}

# Current Visual Observations (8 Directional Views)
The agent is currently observing the environment from 8 directions (clockwise from front). Each image corresponds to:
{direction_names}

IMAGE 1: Front view (0°) - What the agent directly faces
IMAGE 2: Front-Right view (45°) - Diagonal front-right direction  
IMAGE 3: Right view (90°) - Direct right side
IMAGE 4: Back-Right view (135°) - Diagonal back-right direction
IMAGE 5: Back view (180°) - Behind the agent
IMAGE 6: Back-Left view (225°) - Diagonal back-left direction
IMAGE 7: Left view (270°) - Direct left side
IMAGE 8: Front-Left view (315°) - Diagonal front-left direction

# Your Task

Analyze the 8-directional observations to understand the spatial environment, then output a JSON defining the next navigation subtask.

**Core Requirements:**
1. **Spatial Understanding**: Determine your facing direction, identify obstacles/landmarks, estimate distances
2. **Path Planning**: Provide guidance-based navigation (e.g., "move along the wall", "go around the table", "turn at the door")
3. **Subtask Design**: Choose an achievable intermediate destination, ensure it's visible/inferable
4. **Logical Consistency**: Ensure all reasoning parts are coherent - current_location must match spatial perception, navigation guidance must align with destination selection, and all descriptions must be consistent with each other

**Output Quality**: Provide concise guidance that captures key spatial relationships without over-specification.

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (7 Required Fields):**
{{
    "current_location": "Your position + facing direction + nearby landmarks",
    "instruction_sequence": "Complete waypoint chain: A → B → C → D (final goal)",
    "subtask_destination": "The NEXT immediate waypoint",
    "subtask_instruction": "Concise guidance to reach subtask_destination",
    "planning_hints": "**3 parts**: (1) Current facing direction, (2) Destination's relative position, (3) Navigation guidance (e.g., 'move along the wall', 'go around the table', 'turn at the doorway')",
    "completion_criteria": "**3 constraints**: (1) Location description, (2) Key visible objects, (3) Spatial relationships",
    "reasoning": "**3-part analysis**: (1) Spatial perception: facing direction + key obstacles/landmarks, (2) Destination selection: why this waypoint + relative position, (3) Navigation guidance: high-level path strategy. **CRITICAL**: Ensure all parts are logically consistent - facing direction in (1) must match current_location, destination in (2) must match subtask_destination, and guidance in (3) must align with planning_hints"
}}

# Examples for Understanding

## Example 1: Living Room to Kitchen Task
**Global Instruction**: "Go to the kitchen and stop next to the refrigerator"
**Current Observation**: Agent in living room center, kitchen doorway visible in IMAGE 2 at 45° right

**Correct Output**:
{{
    "current_location": "Living room center, facing north toward TV wall",
    "instruction_sequence": "Living room (current) → Kitchen doorway → Kitchen interior → Refrigerator (final)",
    "subtask_destination": "Kitchen doorway entrance",
    "subtask_instruction": "Turn toward the doorway and move forward to reach the entrance",
    "planning_hints": "Currently facing TV wall (north). Kitchen doorway is at your front-right. Navigate by turning right to face the doorway, then move forward along the side of the coffee table to reach the doorway threshold.",
    "completion_criteria": "At doorway threshold, doorway frame visible, close to entrance",
    "reasoning": "(1) Spatial perception: In living room center facing TV wall - kitchen doorway visible at front-right, coffee table between current position and doorway. (2) Destination selection: Kitchen doorway is the first waypoint to reach kitchen area, located at front-right position. (3) Navigation guidance: Turn right to face doorway direction, move forward while avoiding the coffee table, stop at doorway threshold."
}}

## Example 2: Hallway to Bedroom Task  
**Global Instruction**: "Walk down the hallway, enter the bedroom on the right, and stop beside the bed"
**Current Observation**: Agent in hallway start, bedroom door visible in IMAGE 2 at 45° right

**Correct Output**:
{{
    "current_location": "Hallway entrance, facing south down the corridor",
    "instruction_sequence": "Hallway start (current) → Bedroom doorway on right → Bedroom interior → Bed (final)",
    "subtask_destination": "Bedroom doorway on the right",
    "subtask_instruction": "Turn toward the doorway and move forward along the hallway to reach the entrance",
    "planning_hints": "Currently facing south down the hallway. Bedroom doorway is at your front-right on the right wall. Navigate by turning right to face the doorway, then move forward along the right side of the hallway to reach the doorway threshold.",
    "completion_criteria": "At bedroom doorway threshold, doorway frame visible, hallway continues behind",
    "reasoning": "(1) Spatial perception: At hallway entrance facing south - bedroom doorway visible at front-right on the right wall. (2) Destination selection: Bedroom doorway is the first waypoint to enter the bedroom, located at front-right on the hallway wall. (3) Navigation guidance: Turn right to face the doorway on the right wall, move forward along the hallway keeping the right wall as reference, stop at doorway threshold."
}}

# Key Rules
- **Actions**: MOVE_FORWARD (0.25m/step), TURN_LEFT/RIGHT (45°/step), STOP
- **Guidance style**: Use high-level directions (e.g., "move along the wall", "go around the table", "turn at the door") instead of precise step counts
- **IMAGE references**: Use in reasoning only (e.g., "IMAGE 1 shows..."), NOT in other fields
- **Concise output**: Keep descriptions focused on key spatial relationships, avoid redundant details
- **Logical consistency**: ALL output fields must be coherent - no contradictions between current_location, planning_hints, reasoning parts, or any other fields
"""


# Verification and Replanning prompt - combined module for subtask completion verification and next subtask generation
VERIFICATION_REPLANNING_PROMPT = """You are the verification and replanning module for Vision-Language Navigation. Your role is to verify subtask completion and plan the next action.

# Overall Navigation Task Instruction (Global Task)
{instruction}

# Current Subtask Context (from previous planning)
**Subtask Destination**: {subtask_destination}
**Subtask Instruction**: {subtask_instruction}
**Completion Criteria**: {completion_criteria}

# Current Visual Observations (8 Directions)
{direction_names}
IMAGE 1-8: Front(0°), Front-Right(45°), Right(90°), Back-Right(135°), Back(180°), Back-Left(225°), Left(270°), Front-Left(315°)

# Your Task

Analyze current observations and determine if the subtask is completed based on the completion criteria:

**If Subtask IS COMPLETED** (reached subtask destination):
- Set is_completed = true
- Generate the NEXT subtask based on current observations and global goal
- Output all 7 fields (same format as initial planning) PLUS is_completed field

**If Subtask NOT COMPLETED** (destination not reached):
- Set is_completed = false  
- KEEP the same subtask_destination
- MODIFY the subtask_instruction to be more targeted based on current observations
- Output all 7 fields with updated instruction PLUS is_completed field

**Core Requirements:**
1. Use completion criteria as verification standard
2. Analyze spatial relationships and object detection
3. If completed → plan next waypoint in global instruction sequence
4. If not completed → refine instruction for current destination
5. Always provide 4-part reasoning including completion analysis
6. **Logical Consistency**: Ensure reasoning is coherent - spatial perception must support completion check, destination selection must align with is_completed status, and all fields must be internally consistent without contradictions

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (8 Required Fields - 7 from planning + is_completed):**
{{
    "is_completed": true/false,
    "current_location": "Your position + facing direction + nearby landmarks",
    "instruction_sequence": "Complete waypoint chain: A → B → C → D (final goal)",
    "subtask_destination": "If completed: NEXT waypoint. If not completed: SAME as input",
    "subtask_instruction": "If completed: guidance to NEXT destination. If not completed: REFINED guidance to SAME destination",
    "planning_hints": "**3 parts**: (1) Current facing direction, (2) Destination's relative position, (3) Navigation guidance",
    "completion_criteria": "**3 constraints**: (1) Location description, (2) Key visible objects, (3) Spatial relationships",
    "reasoning": "**4-part analysis**: (1) Spatial perception: facing + key landmarks, (2) Destination selection: why + position, (3) Navigation guidance: high-level path strategy, (4) Completion check: verify criteria → conclusion. **CRITICAL**: Maintain logical consistency - completion check (4) must reference spatial perception in (1), is_completed status must match completion check conclusion, and destination in (2) must align with is_completed (NEXT if true, SAME if false)"
}}

# Examples for Understanding

## Example 1: Subtask COMPLETED - Generate Next Subtask
**Input Context**:
- Global Instruction: "Go to kitchen and stop next to refrigerator"
- Current Subtask Destination: "Kitchen doorway entrance"
- Current Subtask Instruction: "Turn right and move forward to kitchen doorway entrance"
- Completion Criteria: "At doorway threshold, doorway frame visible, distance <1m"
- Current Observation: IMAGE 1 shows doorway at 0.5m, can see kitchen interior

**Correct Output**:
{{
    "is_completed": true,
    "current_location": "Kitchen doorway threshold, facing north into kitchen interior",
    "instruction_sequence": "Living room → Kitchen doorway (completed) → Kitchen interior → Refrigerator (final)",
    "subtask_destination": "Kitchen interior center",
    "subtask_instruction": "Move forward into the kitchen interior toward the refrigerator area",
    "planning_hints": "Currently at doorway threshold facing into kitchen. Kitchen interior is directly ahead. Navigate by moving forward through the doorway, look for refrigerator on the right side.",
    "completion_criteria": "Inside kitchen interior, counters and appliances visible, doorway behind",
    "reasoning": "(1) Spatial perception: At doorway threshold facing into kitchen - can see kitchen counters and appliances ahead. (2) Destination selection: Kitchen doorway completed, next is kitchen interior center before approaching refrigerator. (3) Navigation guidance: Move forward through doorway into kitchen interior, scan for refrigerator on right side. (4) Completion check: At doorway threshold ✓, doorway frame visible ✓, close to entrance ✓. All criteria satisfied → is_completed=true."
}}

## Example 2: Subtask NOT COMPLETED - Refine Current Instruction
**Input Context**:
- Global Instruction: "Go to kitchen and stop next to refrigerator"
- Current Subtask Destination: "Kitchen doorway entrance"
- Current Subtask Instruction: "Turn right and move forward to kitchen doorway entrance"
- Completion Criteria: "At doorway threshold, doorway frame visible, distance <1m"
- Current Observation: IMAGE 1 shows coffee table blocking path, IMAGE 2 shows doorway at 45° right, ~4m away

**Correct Output**:
{{
    "is_completed": false,
    "current_location": "Living room, facing north toward TV wall, coffee table in path",
    "instruction_sequence": "Living room (current) → Kitchen doorway → Kitchen interior → Refrigerator (final)",
    "subtask_destination": "Kitchen doorway entrance",
    "subtask_instruction": "Turn toward the doorway and navigate around the coffee table to reach the entrance",
    "planning_hints": "Currently facing north. Kitchen doorway is at front-right. Navigate by turning right to face doorway, then move forward along the side of the coffee table to reach the doorway threshold.",
    "completion_criteria": "At doorway threshold, doorway frame visible, close to entrance",
    "reasoning": "(1) Spatial perception: In living room facing TV wall - coffee table blocks direct path, kitchen doorway visible at front-right. (2) Destination selection: Still targeting kitchen doorway entrance, located at front-right position. (3) Navigation guidance: Turn right to face doorway, move forward while avoiding the coffee table on the left side. (4) Completion check: NOT at doorway threshold ✗, doorway visible but far ✗. Criteria not satisfied → is_completed=false, need to navigate around obstacle."
}}

# Key Rules
- **Actions**: MOVE_FORWARD (0.25m/step), TURN_LEFT/RIGHT (45°/step), STOP
- **Guidance style**: Use high-level directions (e.g., "move along the wall", "go around the table", "turn at the door") instead of precise step counts
- **IMAGE references**: Use in reasoning only, NOT in other fields
- **Completion logic**: is_completed=true only if ALL criteria satisfied
- **Destination continuity**: If not completed, subtask_destination MUST stay same as input
- **Concise output**: Focus on key spatial relationships, avoid redundant details
- **Logical consistency**: ALL reasoning parts and output fields must be coherent and consistent - no contradictions between spatial perception, completion check, is_completed status, or destination selection
"""

# Task completion prompt - checks if entire navigation task is completed
TASK_COMPLETION_PROMPT = """Verify if the ENTIRE navigation task has been completed.

# Navigation Goal
{instruction}

# Current Observations (8 Directions)
{direction_names}
IMAGE 1-8: Front(0°), Front-Right(45°), Right(90°), Back-Right(135°), Back(180°), Back-Left(225°), Left(270°), Front-Left(315°)

# Your Task
Compare current observations with the goal's target location → determine completion confidence.

**CRITICAL**: Ensure logical consistency - confidence score must match analysis content, task_completed status must align with confidence level, and recommendation must be consistent with both.

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (4 Required Fields):**
{{
    "task_completed": true/false,
    "confidence": 0.0-1.0,
    "analysis": "Concise check for goal elements with IMAGE citations",
    "recommendation": "If NOT completed: brief next steps. If completed: null"
}}

**Confidence Scale**:
- 0.9-1.0: All goal elements present, position matches
- 0.7-0.9: Most elements match, minor adjustment needed
- 0.5-0.7: Partial match, significant differences
- < 0.5: Location doesn't match goal

**Key Rules**: 
- Be conservative - only mark completed if confidence > 0.85
- **Logical consistency**: task_completed must align with confidence (true only if >0.85), analysis must support the confidence score, and recommendation must match task_completed status (null if true, steps if false)
"""


def get_initial_planning_prompt(instruction: str, direction_names: list) -> str:
    """
    Get initial planning prompt
    
    Args:
        instruction: Complete navigation instruction
        direction_names: List of direction names
        
    Returns:
        Formatted prompt string
    """
    return INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        direction_names=', '.join(direction_names)
    )


def get_verification_replanning_prompt(instruction: str,
                                      subtask_destination: str,
                                      subtask_instruction: str,
                                      completion_criteria: str,
                                      direction_names: list) -> str:
    """
    Get verification and replanning prompt
    
    Args:
        instruction: Complete navigation instruction (global goal)
        subtask_destination: Current subtask destination
        subtask_instruction: Current subtask instruction
        completion_criteria: Completion criteria
        direction_names: List of direction names
        
    Returns:
        Formatted prompt string
    """
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        completion_criteria=completion_criteria,
        direction_names=', '.join(direction_names)
    )


def get_task_completion_prompt(instruction: str, direction_names: list) -> str:
    """
    Get task completion check prompt
    
    Args:
        instruction: Complete navigation instruction
        direction_names: List of direction names
        
    Returns:
        Formatted prompt string
    """
    return TASK_COMPLETION_PROMPT.format(
        instruction=instruction,
        direction_names=', '.join(direction_names)
    )
