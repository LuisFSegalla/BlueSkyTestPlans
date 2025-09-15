from pathlib import Path

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from dodal.common.beamlines.beamline_utils import (
    device_factory,
    get_path_provider,
    set_path_provider,
)
from dodal.common.visit import (
    LocalDirectoryServiceClient,
    StaticVisitPathProvider,
)
from dodal.plan_stubs.data_session import attach_data_session_metadata_decorator
from dodal.utils import BeamlinePrefix, get_beamline_name
from scanspec.specs import Fly, Line

from ophyd_async.core import DetectorTrigger, StandardFlyer, TriggerInfo
from ophyd_async.epics.motor import Motor
from ophyd_async.epics.pmac import (
    PmacIO,
    PmacTrajectoryTriggerLogic,
)
from ophyd_async.fastcs.panda import (
    HDFPanda,
    SeqTable,
    SeqTableInfo,
    SeqTrigger,
    StaticSeqTableTriggerLogic,
)
from ophyd_async.plan_stubs import ensure_connected

# get_beamline_name with no arguments to get the
# default BL name (from $BEAMLINE)
BL = get_beamline_name("p99")
PREFIX = BeamlinePrefix(BL)


@device_factory()
def panda() -> HDFPanda:
    return HDFPanda(
        f"{PREFIX.beamline_prefix}-MO-PANDA-01:",
        path_provider=get_path_provider(),
        name="panda",
    )


set_path_provider(
    StaticVisitPathProvider(
        BL,
        Path("/dls/p99/data/2025/cm40656-4/tmp/"),
        client=LocalDirectoryServiceClient(),
    )
)


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


def panda_scan(start: float, stop: float, num: int, duration: float):
    p = panda()
    motor_x = Motor(prefix="BL99P-MO-STAGE-02:X", name="Motor_X")
    motor_y = Motor(prefix="BL99P-MO-STAGE-02:Y", name="Motor_Y")
    pmac = PmacIO(
        prefix="BL99P-MO-STEP-01:",
        raw_motors=[motor_y, motor_x],
        coord_nums=[1],
        name="pmac",
    )
    yield from ensure_connected(pmac, motor_x, motor_y, p)
    panda_seq = StandardFlyer(StaticSeqTableTriggerLogic(p.seq[1]))

    # Prepare motor info using trajectory scanning
    spec = Fly(
        float(duration)
        @ (Line(motor_y, start, stop, num) * Line(motor_x, start, stop, num))
    )

    trigger_logic = spec
    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    motor_x_mres = -2e-05

    table = SeqTable()
    positions = [int(x / motor_x_mres) for x in spec.frames().lower[motor_x]]

    direction = (
        SeqTrigger.POSA_LT
        if start * motor_x_mres > stop * motor_x_mres
        else SeqTrigger.POSA_GT
    )

    counter = 0
    for pos in positions:
        if counter == num:
            table += SeqTable.row(
                repeats=1,
                trigger=SeqTrigger.BITA_0,
            )
            table += SeqTable.row(
                repeats=1,
                trigger=SeqTrigger.BITA_1,
            )
            counter = 0

        table += SeqTable.row(
            repeats=1,
            trigger=direction,
            position=pos,
            time1=1,
            outa1=True,
            time2=1,
            outa2=False,
        )

        counter += 1
    seq_table_info = SeqTableInfo(sequence_table=table, repeats=1, prescale_as_us=1)

    # Prepare Panda file writer trigger info
    panda_hdf_info = TriggerInfo(
        number_of_events=num * num,
        trigger=DetectorTrigger.CONSTANT_GATE,
        livetime=duration,
        deadtime=1e-5,
    )

    @attach_data_session_metadata_decorator()
    @bpp.run_decorator()
    @bpp.stage_decorator([p, panda_seq])
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, trigger_logic, wait=True)
        # prepare sequencer table
        yield from bps.prepare(panda_seq, seq_table_info, wait=True)
        # prepare panda and hdf writer once, at start of scan
        yield from bps.prepare(p, panda_hdf_info, wait=True)

        # kickoff devices waiting for all of them
        yield from bps.kickoff(p, wait=True)
        yield from bps.kickoff(panda_seq, wait=True)
        yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

        yield from bps.complete_all(pmac_trajectory_flyer, p, panda_seq, wait=True)

    yield from inner_plan()
