import Assumptions
import BandOccupancy
import BarrierRescaling
import CouplingPower
import CouplingSign
import CrossCovIndep
import CrossCov
import CrossCovProbe
import CrossCovSLLN
import DiffusionOrdering
import FreeDiffusion
import FWReduction
import InversePSDAntitone
import KramersBridge
import KramersExitTime
import LogDensityCurvature
import MinibatchNoise
import PgdBaseline
import PowerOfS
import ProjectionClamp
import PullbackHessian
import QuasipotentialFTC
import QuasipotentialMonotone
import ShoulderAttenuation
import ShoulderDwell
import SoftplusBand
import StationaryDensity
import StructuralZeros
import TaylorChain
import TraceLemmas
import UpdateCovariance
import UpdateDrift
import WitnessInstance

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let ours : Array Name := #[
    `Assumptions,
    `BandOccupancy,
    `BarrierRescaling,
    `CouplingPower,
    `CouplingSign,
    `CrossCovIndep,
    `CrossCov,
    `CrossCovProbe,
    `CrossCovSLLN,
    `DiffusionOrdering,
    `FreeDiffusion,
    `FWReduction,
    `InversePSDAntitone,
    `KramersBridge,
    `KramersExitTime,
    `LogDensityCurvature,
    `MinibatchNoise,
    `PgdBaseline,
    `PowerOfS,
    `ProjectionClamp,
    `PullbackHessian,
    `QuasipotentialFTC,
    `QuasipotentialMonotone,
    `ShoulderAttenuation,
    `ShoulderDwell,
    `SoftplusBand,
    `StationaryDensity,
    `StructuralZeros,
    `TaylorChain,
    `TraceLemmas,
    `UpdateCovariance,
    `UpdateDrift,
    `WitnessInstance,
    `IcnnLift]
  let mut audited := 0
  let mut thms := 0
  let mut declaredAxioms := 0
  let mut internal := 0
  let mut bad := 0
  let mut sorries := 0
  let mut badNames : Array Name := #[]
  let mut sorryNames : Array Name := #[]
  for (n, ci) in env.constants.toList do
    let isThm := match ci with | .thmInfo _ => true | _ => false
    let isAx := match ci with | .axiomInfo _ => true | _ => false
    if true then
      let some idx := env.getModuleIdxFor? n | continue
      let some mn := env.header.moduleNames[idx.toNat]? | continue
      if !(ours.contains mn) then continue
      if isAx then declaredAxioms := declaredAxioms + 1
      if n.isInternal then internal := internal + 1
      if isThm then thms := thms + 1
      audited := audited + 1
      let ax ← Lean.collectAxioms n
      for a in ax do
        if a == ``sorryAx then
          sorries := sorries + 1
          sorryNames := sorryNames.push n
        else if a != ``propext && a != ``Classical.choice && a != ``Quot.sound then
          bad := bad + 1
          badNames := badNames.push n
  logInfo m!"AUDITED_DECLARATIONS {audited}"
  logInfo m!"OF_WHICH_THEOREMS {thms}"
  logInfo m!"DECLARED_AXIOMS_OF_OURS {declaredAxioms}"
  logInfo m!"INTERNAL_OF_THOSE {internal}"
  logInfo m!"NONSTANDARD_AXIOM_HITS {bad}"
  logInfo m!"SORRYAX_HITS {sorries}"
  logInfo m!"NONSTANDARD_NAMES {badNames}"
  logInfo m!"SORRY_NAMES {sorryNames}"
