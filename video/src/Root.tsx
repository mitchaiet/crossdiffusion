import {Composition} from 'remotion';
import {PublicOverview} from './PublicOverview';

export const Root = () => (
  <Composition
    id="CrossDiffusionPublicOverview"
    component={PublicOverview}
    durationInFrames={1800}
    fps={30}
    width={1920}
    height={1080}
  />
);
