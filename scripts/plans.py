import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

# import h5py
from aioca import caput
from bluesky.utils import MsgGenerator
from dodal.plan_stubs.data_session import attach_data_session_metadata_decorator
from scanspec.specs import Fly, Line

from ophyd_async.core import (
    DetectorTrigger,
    FlyMotorInfo,
    StandardFlyer,
    TriggerInfo,
    wait_for_value,
)
from ophyd_async.epics.motor import Motor
from ophyd_async.epics.pmac import (
    PmacIO,
    PmacTrajectoryTriggerLogic,
)
from ophyd_async.fastcs.panda import (
    HDFPanda,
    PandaPcompDirection,
    PcompInfo,
    SeqTable,
    SeqTableInfo,
    SeqTrigger,
    StaticPcompTriggerLogic,
    StaticSeqTableTriggerLogic,
)
from ophyd_async.fastcs.panda._block import PcompBlock
from ophyd_async.plan_stubs import ensure_connected


def no_panda():
    # Defining the frlyers and components of the scan
    motor_x = Motor(prefix="BL99P-MO-STAGE-02:X", name="Motor_X")
    motor_y = Motor(prefix="BL99P-MO-STAGE-02:Y", name="Motor_Y")
    pmac = PmacIO(
        prefix="BL99P-MO-STEP-01:",
        raw_motors=[motor_y, motor_x],
        coord_nums=[1],
        name="pmac",
    )

    yield from ensure_connected(pmac, motor_x, motor_y)

    # Prepare motor info using trajectory scanning
    spec = Fly(float(1) @ (Line(motor_y, 0, 1, 10) * Line(motor_x, 0, 1, 10)))

    trigger_logic = spec
    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    @bpp.run_decorator()
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, trigger_logic, wait=True)

        # kickoff devices waiting for all of them
        yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

        yield from bps.complete_all(pmac_trajectory_flyer, wait=True)

    yield from inner_plan()
