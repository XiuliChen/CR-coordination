EYE,HAND=0,1
PREP,RAMPUP,RAMPDOWN,FIXATE,STILL=-1,-0.5,0.5,1,1
# hand STILL, eye FIXATE

NEW_COMMAND=-1
NO_OP=0
CLICK=1


SCALE_DEG=40


'''
self.actions=np.array([ [NO_OP,NO_OP],
                        [NEW_COMMAND,NO_OP],
                        [NO_OP,NEW_COMMAND],
                        [NO_OP,CLICK],
                        ])
'''

ACTION_NOOP=0
ACTION_NEW_EYE_COMMAND=1
ACTION_NEW_HAND_COMMAND=2
ACTION_CLICK=3
