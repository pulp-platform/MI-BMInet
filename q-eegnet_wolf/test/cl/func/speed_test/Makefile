PULP_APP = test

PULP_APP_CL_SRCS = \
    cluster.c \

PULP_APP_FC_SRCS = \
    test.c \

PULP_LDFLAGS += -lplpdsp
PULP_CFLAGS = -O3 -g 

PULP_CFLAGS += -DMULT=True

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
