import asyncio
from datetime import datetime
from pathlib import Path
import math as mt
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import h5py
import numpy as np
from aioca import caput
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
from scanspec.specs import Array, Fly, Line

from ophyd_async.core import DetectorTrigger, StandardFlyer, TriggerInfo, wait_for_value
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
    ScanSpecInfo,
    ScanSpecSeqTableTriggerLogic,
    PosOutScaleOffset
)
from ophyd_async.fastcs.panda._block import PcompBlock
from ophyd_async.plan_stubs import ensure_connected

# get_beamline_name with no arguments to get the
# default BL name (from $BEAMLINE)
BL = get_beamline_name("p99")
PREFIX = BeamlinePrefix(BL)


class _StaticPcompTriggerLogic(StaticPcompTriggerLogic):
    """For controlling the PandA `PcompBlock` when flyscanning."""

    def __init__(self, pcomp: PcompBlock) -> None:
        self.pcomp = pcomp

    async def kickoff(self) -> None:
        await wait_for_value(self.pcomp.active, True, timeout=1)

    async def prepare(self, value: PcompInfo) -> None:
        await caput("BL99P-MO-PANDA-01:SRGATE1:FORCE_RST", "1", wait=True)
        await asyncio.gather(
            self.pcomp.start.set(value.start_postion),
            self.pcomp.width.set(value.pulse_width),
            self.pcomp.step.set(value.rising_edge_step),
            self.pcomp.pulses.set(value.number_of_pulses),
            self.pcomp.dir.set(value.direction),
        )

    async def stop(self):
        pass


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
        Path("/dls/p99/data/2025/cm40656-5/"),
        client=LocalDirectoryServiceClient(),
    )
)

PATH = "/dls/p99/data/2025/cm40656-5/"


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
    spec = Fly(0.1 @ (Line(motor_y, 0, 1, 10) * ~Line(motor_x, -2, 2, 1000)))

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
        number_of_events=num * (num - 1),
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


def array_scan(start: float, stop: float, num: int, duration: float):
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

    low = np.append(np.linspace(-2, 0, 100), np.linspace(1, 6, 100))
    diff = np.append(np.repeat(low[1] - low[0], 10), np.repeat(low[-1] - low[-2], 100))
    up = low + diff

    # Prepare motor info using trajectory scanning
    spec = Fly(
        float(duration)
        @ Array(axis=motor_x, _upper=up, _lower=low)
    )

    # spec = Fly(
    #     float(duration)
    #     @ Array(axis=motor_x, _midpoints=array, _gap=gaps)
    # )

    trigger_logic = spec
    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    motor_x_mres = -2e-05

    table = SeqTable()
    positions = [int(x / motor_x_mres) for x in spec.frames().lower[motor_x]]

    # Writes down the desired positions that will be written to the sequencer table
    f = h5py.File(
        f"{PATH}p99-extra-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.h5", "w"
    )
    f.create_dataset("positions", shape=(1, len(positions)), data=spec.frames().lower[motor_x])
    f.create_dataset("gaps", shape=(1, len(spec.frames().gap)), data=spec.frames().gap)

    # direction = (
    #     SeqTrigger.POSA_LT
    #     if start * motor_x_mres > stop * motor_x_mres
    #     else SeqTrigger.POSA_GT
    # )

    direction = SeqTrigger.POSA_LT

    table += SeqTable.row(
        repeats=1,
        trigger=SeqTrigger.BITA_0,
    )
    table += SeqTable.row(
        repeats=1,
        trigger=SeqTrigger.BITA_1,
    )

    counter = 0
    for pos in positions:
        if counter == len(low):
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
            outb1=True,
            time2=1,
            outa2=False,
            outb2=True,
        )

        counter += 1
    seq_table_info = SeqTableInfo(sequence_table=table, repeats=1, prescale_as_us=1)

    # Prepare Panda file writer trigger info
    panda_hdf_info = TriggerInfo(
        number_of_events=len(low),
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


def pcomp_scan(start: float, stop: float, num: int, duration: float):
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
    panda_pcomp1 = StandardFlyer(_StaticPcompTriggerLogic(p.pcomp[1]))
    panda_pcomp2 = StandardFlyer(_StaticPcompTriggerLogic(p.pcomp[2]))

    # Prepare motor info using trajectory scanning
    spec = Fly(
        float(duration)
        @ (Line(motor_y, start, stop, num) * ~Line(motor_x, start, stop, num))
    )

    trigger_logic = spec
    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    motor_x_mres = -2e-05

    width = (stop - start) / (num - 1)

    direction_pcomp1 = (
        PandaPcompDirection.NEGATIVE
        if start * motor_x_mres > stop * motor_x_mres
        else PandaPcompDirection.POSITIVE
    )
    direction_pcomp2 = (
        PandaPcompDirection.POSITIVE
        if direction_pcomp1 == PandaPcompDirection.NEGATIVE
        else PandaPcompDirection.NEGATIVE
    )

    pcomp1_info = PcompInfo(
        start_postion=mt.floor(start / motor_x_mres),
        pulse_width=1,
        rising_edge_step=mt.ceil(abs(width / motor_x_mres)),
        number_of_pulses=num,
        direction=direction_pcomp1
    )
    pcomp2_info = PcompInfo(
        start_postion=mt.floor(stop / motor_x_mres),
        pulse_width=1,
        rising_edge_step=mt.ceil(abs(width / motor_x_mres)),
        number_of_pulses=num,
        direction=direction_pcomp2
    )
    # Prepare Panda file writer trigger info
    panda_hdf_info = TriggerInfo(
        number_of_events=num * num,
        trigger=DetectorTrigger.CONSTANT_GATE,
        livetime=duration,
        deadtime=1e-5,
    )

    @attach_data_session_metadata_decorator()
    @bpp.run_decorator()
    @bpp.stage_decorator([p, panda_pcomp1, panda_pcomp2])
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, trigger_logic, wait=True)
        # prepare sequencer table
        yield from bps.prepare(panda_pcomp1, pcomp1_info, wait=True)
        yield from bps.prepare(panda_pcomp2, pcomp2_info, wait=True)
        # prepare panda and hdf writer once, at start of scan
        yield from bps.prepare(p, panda_hdf_info, wait=True)

        # kickoff devices waiting for all of them
        yield from bps.kickoff(p, wait=True)
        yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

        yield from bps.complete_all(pmac_trajectory_flyer, p, wait=True)

    yield from inner_plan()


def time_based_scan(start: float, stop: float, num: int, duration: float):
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

    # Prepare motor info using trajectory scanning
    spec = Fly(
        float(duration)
        @ (Line(motor_y, start, stop, num) * Line(motor_x, start, stop, num))
    )

    info = ScanSpecInfo(spec=spec, deadtime=1 * 10**-6)
    seq_table_flyer = ScanSpecSeqTableTriggerLogic(
        p.seq[1],
        {
            motor_x: PosOutScaleOffset(
                "INENC2.VAL",
                p.inenc[2].val_scale,
                p.inenc[2].val_offset,
            ),  # type: ignore
            motor_y: PosOutScaleOffset(
                "INENC3.VAL",
                p.inenc[3].val_scale,
                p.inenc[3].val_offset,
            ),  # type: ignore
        },
    )
    seq_table_flyer = StandardFlyer(seq_table_flyer)

    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    # Prepare Panda file writer trigger info
    panda_hdf_info = TriggerInfo(
        number_of_events=num * num,
        trigger=DetectorTrigger.CONSTANT_GATE,
        livetime=duration,
        deadtime=1e-5,
    )

    @attach_data_session_metadata_decorator()
    @bpp.run_decorator()
    @bpp.stage_decorator([p, seq_table_flyer])
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, spec, wait=True)
        # prepare sequencer table
        yield from bps.prepare(seq_table_flyer, info, wait=True)
        # prepare panda and hdf writer once, at start of scan
        yield from bps.prepare(p, panda_hdf_info, wait=True)

        # kickoff devices waiting for all of them
        yield from bps.kickoff(p, wait=True)
        yield from bps.kickoff(seq_table_flyer, wait=True)
        yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

        yield from bps.complete_all(pmac_trajectory_flyer, p, seq_table_flyer, wait=True)

    yield from inner_plan()