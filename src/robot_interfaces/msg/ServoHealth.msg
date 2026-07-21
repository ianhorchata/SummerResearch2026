# Per-servo health snapshot, published periodically by arm_node.
std_msgs/Header header
int32[]   ids              # servo ids discovered on the bus
float64[] voltage          # volts, parallel to ids
int32[]   temperature      # degrees C, parallel to ids
bool[]    torque_on        # whether holding torque is engaged
string[]  issues           # human-readable issue summary per servo ("" = healthy)
