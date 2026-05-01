import asyncio
import math as mt
from datetime import datetime
from itertools import pairwise
from pathlib import Path

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
# from scanspec.specs import Array, Fly, Line
from scanspec.specs import Fly, Line

from ophyd_async.core import DetectorTrigger, StandardFlyer, TriggerInfo, wait_for_value
from ophyd_async.epics.motor import Motor
from ophyd_async.epics.pmac import PmacIO, PmacScanInfo, PmacTrajectoryTriggerLogic
from ophyd_async.fastcs.panda import (
    HDFPanda,
    PandaPcompDirection,
    PcompInfo,
    PosOutScaleOffset,
    ScanSpecInfo,
    ScanSpecSeqTableTriggerLogic,
    SeqTable,
    SeqTableInfo,
    SeqTrigger,
    StaticPcompTriggerLogic,
    StaticSeqTableTriggerLogic,
)
from ophyd_async.fastcs.panda._block import PcompBlock
from ophyd_async.plan_stubs import ensure_connected
from ophyd_async.fastcs.xspress import XspressDetector, XspressTriggerInfo
# get_beamline_name with no arguments to get the
# default BL name (from $BEAMLINE)
BL = get_beamline_name("P51")
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
        "BL51P-EA-PANDA-02:",
        path_provider=get_path_provider(),
        name="panda",
    )

@device_factory()
def panda_encoder() -> HDFPanda:
    """Main PandA with the motor encoders."""
    return HDFPanda(
        "BL51P-EA-PANDA-02:",
        path_provider=get_path_provider(),
        name="panda2",
    )

@device_factory()
def panda_fast() -> HDFPanda:
    """Second PandA without FMC card."""
    return HDFPanda(
        "BL51P-EA-PANDA-01:",
        path_provider=get_path_provider(),
        name="panda1",
    )


def xsp() -> XspressDetector:
    """Xspress detector simulation."""
    return XspressDetector(
        prefix="BL51P-EA-XSP-01:",
        path_provider=get_path_provider(),
        name="xspress",
    )

set_path_provider(
    StaticVisitPathProvider(
        BL,
        Path("/dls/p51/data/2026/cm44254-2/tmp"),
        client=LocalDirectoryServiceClient(),
    )
)

PATH = "/dls/p99/data/2025/cm40656-5/"


def no_panda(start: float,
             stop: float,
             num: int,
             duration: float,
             repetitions: int = 1,
             ramp_time: float | None = None,
             turnaround_time: float | None = None
    ):
    motor_x = Motor(prefix="BL51P-OP-PCHRO-01:TS:XFINE", name="Motor_X")
    pmac = PmacIO(
        prefix="BL51P-MO-STEP-06:",
        raw_motors=[motor_x],
        coord_nums=[3],
        name="pmac",
    )
    yield from ensure_connected(pmac, motor_x)

    # Prepare motor info using trajectory scanning
    spec = Fly(duration @ (repetitions * ~Line(motor_x, start, stop, num)))

    trigger_logic = PmacScanInfo(
        spec=spec,
        ramp_time=ramp_time,
        turnaround_time=turnaround_time
    )
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


def panda_debug(
    start: float,
    stop: float,
    num: int,
    duration: float,
    repetitions: int = 1,
    ramp_time: float | None = None,
    turnaround_time: float | None = None,
):
    p1 = panda_fast()
    p2 = panda_encoder()
    motor_x = Motor(prefix="BL51P-OP-PCHRO-01:TS:XFINE", name="Motor_X")
    pmac = PmacIO(
        prefix="BL51P-MO-STEP-06:",
        raw_motors=[motor_x],
        coord_nums=[3],
        name="pmac",
    )
    yield from ensure_connected(pmac, motor_x, p1, p2)

    panda_seq = StandardFlyer(StaticSeqTableTriggerLogic(p2.seq[1]))

    step = abs((stop - start) / num)

    motor_x_mres = -0.0001

    if repetitions < 2:
        positions = np.arange(
            start,
            stop + 0.5 * step,
            step,
        )
    else:
        positions = np.append(
        np.arange(
            start,
            stop,
            step,),
        np.arange(
            stop,
            start + 0.5 * step,
            -1 * step)
        )

    # For simplicity sake I'll force the repetitions to be either 1 or an even number
    if repetitions != 1 and (repetitions % 2) != 0:
        repetitions = repetitions + 1

    # Prepare motor info using trajectory scanning
    spec = Fly(duration @ (repetitions * ~Line(motor_x, start, stop, num)))

    info = PmacScanInfo(
        spec=spec,
        ramp_time=ramp_time,
        turnaround_time=turnaround_time
    )
    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    panda_hdf_info = TriggerInfo(
        number_of_events=0,
        trigger=DetectorTrigger.EXTERNAL_LEVEL,
        livetime=duration - 1e-5,
        deadtime=1e-5,
    )

    table = SeqTable.empty()

    positions = np.ndarray(2 * num)

    # positions = (positions / motor_x_mres).astype(int)
    tmp = [(x / motor_x_mres).astype(int)
                 for x in spec.frames().lower[motor_x]]
    positions[0:num] = tmp[0:num]
    tmp = [(x / motor_x_mres).astype(int)
        for x in spec.frames().upper[motor_x]]
    positions[num:2 * num] = tmp[num - 1:(2 * num) - 1]

    print(f"positions = {positions}")

    direction = [
        SeqTrigger.POSA_GT if current < next else SeqTrigger.POSA_LT
        for current, next in pairwise(positions)
    ]
    direction.append(direction[-1])

    counter = 0
    for pos, trig in zip(positions, direction, strict=True):
        counter += 1
        if counter - 1 == 0:
            # int(duration / 1e-6) - 1,
            table += SeqTable.row(
                repeats=1,
                trigger=trig,
                position=pos,
                time1=int(duration / 1e-6) - 1,
                outa1=True,
                outb1=True,
                time2=1,
                outa2=False,
                outb2=False
            )

        elif counter == num:
            table += SeqTable.row(
                repeats=1,
                trigger=trig,
                position=pos,
                time1=int(duration / 1e-6) - 1,
                outa1=True,
                outb1=True,
                time2=1,
                outa2=False,
                outb2=False
            )
            counter = 0

        else:
            table += SeqTable.row(
                repeats=1,
                trigger=trig,
                position=pos,
                time1=int(duration / 1e-6) - 1,
                outa1=True,
                time2=1,
                outa2=False,
            )

    seq_table_info = SeqTableInfo(sequence_table=table,
                                  repeats=int(repetitions / 2),
                                  prescale_as_us=1)

    @attach_data_session_metadata_decorator()
    @bpp.run_decorator()
    @bpp.stage_decorator([p1, p2, panda_seq])
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, info, wait=True)

        # prepare sequencer table
        yield from bps.prepare(panda_seq, seq_table_info, wait=True)
        # prepare panda and hdf writer once, at start of scan
        yield from bps.prepare(p1, panda_hdf_info)
        yield from bps.prepare(p2, panda_hdf_info, wait=True)

        # kickoff devices waiting for all of them
        yield from bps.kickoff(p1, wait=True)
        yield from bps.kickoff(p2, wait=True)
        yield from bps.kickoff(panda_seq, wait=True)
        yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

        yield from bps.complete_all(pmac_trajectory_flyer, panda_seq, wait=True)

    yield from inner_plan()



def panda_capture(
        start: float,
        stop: float,
        num: int,
        duration: float,
        repetitions: int = 1,
        ramp_time: float | None = None,
        turnaround_time: float | None = None,
    ):
    p = panda()
    motor_x = Motor(prefix="BL51P-OP-PCHRO-01:TS:XFINE", name="Motor_X")
    pmac = PmacIO(
        prefix="BL51P-MO-STEP-06:",
        raw_motors=[motor_x],
        coord_nums=[3],
        name="pmac",
    )
    yield from ensure_connected(pmac, motor_x, p)

    # Prepare motor info using trajectory scanning
    spec = Fly(duration @ (repetitions * ~Line(motor_x, start, stop, num)))

    trigger_logic = PmacScanInfo(
        spec=spec,
        ramp_time=ramp_time,
        turnaround_time=turnaround_time
    )
    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    panda_hdf_info = TriggerInfo(
        number_of_events=0,
        trigger=DetectorTrigger.EXTERNAL_LEVEL,
        livetime=duration - 1e-5,
        deadtime=1e-5,
    )

    @attach_data_session_metadata_decorator()
    @bpp.run_decorator()
    @bpp.stage_decorator([p])
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, trigger_logic, wait=True)
        yield from bps.prepare(p, panda_hdf_info, wait=True)

        # kickoff devices waiting for all of them
        yield from bps.kickoff(p, wait=True)
        yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

        yield from bps.complete_all(pmac_trajectory_flyer, p, wait=True)

    yield from inner_plan()


def panda_scan(start: float,
               stop: float,
               num: int,
               duration: float,
               repetitions: int = 1,
               ramp_time: float | None = None,
               turnaround_time: float | None = None
    ):
    p = panda()
    motor_x = Motor(prefix="BL51P-OP-PCHRO-01:TS:XFINE", name="Motor_X")
    pmac = PmacIO(
        prefix="BL51P-MO-STEP-06:",
        raw_motors=[motor_x],
        coord_nums=[3],
        name="pmac",
    )
    yield from ensure_connected(pmac, motor_x, p)
    panda_seq = StandardFlyer(StaticSeqTableTriggerLogic(p.seq[1]))

    step = abs((stop - start) / num)

    motor_x_mres = -0.0001
    positions = np.append(np.arange(
        start,
        stop,
        step,
    ), (np.arange(stop, start, -1 * step)))

    # Prepare motor info using trajectory scanning
    spec = Fly(
        float(duration)
        @ (repetitions * ~Line(motor_x, start, stop, num))
    )

    info = PmacScanInfo(
        spec=spec,
        ramp_time=ramp_time,
        turnaround_time=turnaround_time
    )

    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    table = SeqTable.empty()

    positions = (positions / motor_x_mres).astype(int)

    direction = [
        SeqTrigger.POSA_GT if current < next else SeqTrigger.POSA_LT
        for current, next in pairwise(positions)
    ]
    direction.append(direction[-1])
    acq_duration = (num * duration) / (num + 1)
    print(f"acq_duration = {acq_duration}")
    for pos, trig in zip(positions, direction, strict=True):

        table += SeqTable.row(
            repeats=1,
            trigger=trig,
            position=pos,
            time1=int(duration / 1e-6) - 1,
            outa1=True,
            time2=1,
            outa2=False,
        )

    seq_table_info = SeqTableInfo(sequence_table=table,
                                  repeats=int(repetitions / 2),
                                  prescale_as_us=1)

    # Prepare Panda file writer trigger info
    panda_hdf_info = TriggerInfo(
        number_of_events=0,
        trigger=DetectorTrigger.EXTERNAL_LEVEL,
        livetime=duration - 1e-5,
        deadtime=1e-5,
    )

    @attach_data_session_metadata_decorator()
    @bpp.run_decorator()
    @bpp.stage_decorator([p, panda_seq])
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, info, wait=True)
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


def xsp_sim_scan(start: float,
               stop: float,
               num: int,
               duration: float,
               repetitions: int = 1,
               ramp_time: float | None = None,
               turnaround_time: float | None = None
    ):
    p = panda()
    det = xsp()
    motor_x = Motor(prefix="BL51P-OP-PCHRO-01:TS:XFINE", name="Motor_X")
    pmac = PmacIO(
        prefix="BL51P-MO-STEP-06:",
        raw_motors=[motor_x],
        coord_nums=[3],
        name="pmac",
    )
    yield from ensure_connected(pmac, motor_x, p, det)
    panda_seq = StandardFlyer(StaticSeqTableTriggerLogic(p.seq[1]))

    motor_x_mres = -0.0001

    # Prepare motor info using trajectory scanning
    spec = Fly(
        float(duration)
        @ (repetitions * ~Line(motor_x, start, stop, num))
    )

    info = PmacScanInfo(
        spec=spec,
        ramp_time=ramp_time,
        turnaround_time=turnaround_time
    )

    pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
    pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

    table = SeqTable.empty()

    positions = np.ndarray(2 * num)

    # positions = (positions / motor_x_mres).astype(int)
    tmp = [(x / motor_x_mres).astype(int)
                 for x in spec.frames().lower[motor_x]]
    positions[0:num] = tmp[0:num]
    tmp = [(x / motor_x_mres).astype(int)
        for x in spec.frames().upper[motor_x]]
    positions[num:2 * num] = tmp[num - 1:(2 * num) - 1]

    direction = [
        SeqTrigger.POSA_GT if current < next else SeqTrigger.POSA_LT
        for current, next in pairwise(positions)
    ]
    direction.append(direction[-1])
    acq_duration = (num * duration) / (num + 1)
    print(f"acq_duration = {acq_duration}")
    for pos, trig in zip(positions, direction, strict=True):

        table += SeqTable.row(
            repeats=1,
            trigger=trig,
            position=pos,
            time1=int(duration / 1e-6) - 1,
            outa1=True,
            time2=1,
            outa2=False,
        )

    seq_table_info = SeqTableInfo(sequence_table=table,
                                  repeats=int(repetitions / 2),
                                  prescale_as_us=1)

    # Prepare Panda file writer trigger info
    panda_hdf_info = TriggerInfo(
        number_of_events=0,
        trigger=DetectorTrigger.EXTERNAL_LEVEL,
        livetime=duration - 1e-5,
        deadtime=1e-5,
    )

    xsp_trigger = XspressTriggerInfo(
        number_of_events=int(num * repetitions),
        trigger=DetectorTrigger.INTERNAL,
        livetime=duration,
        chunk=int(1 / duration),
    )

    @attach_data_session_metadata_decorator()
    @bpp.run_decorator()
    @bpp.stage_decorator([p, panda_seq, det])
    def inner_plan():
        # Prepare pmac with the trajectory
        yield from bps.prepare(pmac_trajectory_flyer, info, wait=True)
        # prepare sequencer table
        yield from bps.prepare(panda_seq, seq_table_info, wait=True)
        # prepare panda and hdf writer once, at start of scan
        yield from bps.prepare(p, panda_hdf_info, wait=True)
        # prepare xsp sim hdf writer
        yield from bps.prepare(det, xsp_trigger, wait=True)

        # kickoff devices waiting for all of them
        yield from bps.kickoff(p, wait=True)
        yield from bps.kickoff(panda_seq, wait=True)
        yield from bps.kickoff(det, wait=True)
        yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

        yield from bps.complete_all(pmac_trajectory_flyer, p, panda_seq, det, wait=True)

    yield from inner_plan()


# def array_scan(start: float, stop: float, num: int, duration: float):
#     p = panda()
#     motor_x = Motor(prefix="BL99P-MO-STAGE-02:X", name="Motor_X")
#     motor_y = Motor(prefix="BL99P-MO-STAGE-02:Y", name="Motor_Y")
#     pmac = PmacIO(
#         prefix="BL99P-MO-STEP-01:",
#         raw_motors=[motor_y, motor_x],
#         coord_nums=[1],
#         name="pmac",
#     )
#     yield from ensure_connected(pmac, motor_x, motor_y, p)
#     panda_seq = StandardFlyer(StaticSeqTableTriggerLogic(p.seq[1]))

#     rad = np.arange(0, 2 * np.pi, 0.005)
#     deg = 3 * np.cos(rad)

#     # Prepare motor info using trajectory scanning
#     spec = Fly(float(duration) @ (Array(axis=motor_x, array=deg)))

#     trigger_logic = spec
#     pmac_trajectory = PmacTrajectoryTriggerLogic(pmac)
#     pmac_trajectory_flyer = StandardFlyer(pmac_trajectory)

#     motor_x_mres = -2e-05

#     table = SeqTable()
#     positions = [int(x / motor_x_mres) for x in spec.frames().lower[motor_x]]

#     # Writes down the desired positions that will be written to the sequencer table
#     f = h5py.File(
#         f"{PATH}p99-extra-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.h5", "w"
#     )
#     f.create_dataset("positions", shape=(1, len(positions)), data=spec.frames().lower[motor_x])
#     f.create_dataset("gaps", shape=(1, len(spec.frames().gap)), data=spec.frames().gap)

#     direction = [
#         SeqTrigger.POSA_GT if a > b else SeqTrigger.POSA_LT
#         for a, b in zip(
#             spec.frames().lower[motor_x], spec.frames().upper[motor_x], strict=True
#         )
#     ]

#     for di, pos in zip(direction, positions, strict=True):
#         table += SeqTable.row(
#             repeats=1,
#             trigger=di,
#             position=pos,
#             time1=1,
#             outa1=True,
#             time2=1,
#             outa2=False,
#         )

#     seq_table_info = SeqTableInfo(sequence_table=table, repeats=1, prescale_as_us=1)

#     # Prepare Panda file writer trigger info
#     panda_hdf_info = TriggerInfo(
#         number_of_events=len(deg),
#         trigger=DetectorTrigger.CONSTANT_GATE,
#         livetime=duration,
#         deadtime=1e-5,
#     )

#     @attach_data_session_metadata_decorator()
#     @bpp.run_decorator()
#     @bpp.stage_decorator([p, panda_seq])
#     def inner_plan():
#         # Prepare pmac with the trajectory
#         yield from bps.prepare(pmac_trajectory_flyer, trigger_logic, wait=True)
#         # prepare sequencer table
#         yield from bps.prepare(panda_seq, seq_table_info, wait=True)
#         # prepare panda and hdf writer once, at start of scan
#         yield from bps.prepare(p, panda_hdf_info, wait=True)

#         # kickoff devices waiting for all of them
#         yield from bps.kickoff(p, wait=True)
#         yield from bps.kickoff(panda_seq, wait=True)
#         yield from bps.kickoff(pmac_trajectory_flyer, wait=True)

#         yield from bps.complete_all(pmac_trajectory_flyer, p, panda_seq, wait=True)

#     yield from inner_plan()


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