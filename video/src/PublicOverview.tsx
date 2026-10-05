import React from 'react';
import {AbsoluteFill, Audio, interpolate, staticFile, useCurrentFrame} from 'remotion';

const ink = '#20352e';
const green = '#3d7657';
const mint = '#dcebdd';
const cream = '#f6f5ee';
const amber = '#dcaa54';
const scenes = [
  {start: 0, end: 270, number: '01', title: 'Bring a puzzle', detail: 'Photo or PDF', action: 'Upload'},
  {start: 270, end: 570, number: '02', title: 'Read the page', detail: 'Grid and clue text', action: 'Read'},
  {start: 570, end: 900, number: '03', title: 'Check the transcription', detail: 'Edit a square or clue', action: 'Check'},
  {start: 900, end: 1290, number: '04', title: 'Fill a first draft', detail: 'Masked letters, several passes', action: 'Fill'},
  {start: 1290, end: 1620, number: '05', title: 'Recheck with the LLM', detail: 'Clues and crossing letters', action: 'Review'},
  {start: 1620, end: 1800, number: '06', title: 'Inspect the result', detail: 'Draft and changes stay visible', action: 'Result'},
];
const blocked = new Set([3, 11, 19, 24, 28, 36, 44, 52, 56, 61, 69, 77]);

const fade = (frame: number, start: number, end: number) =>
  interpolate(frame, [start, start + 16, end - 16, end], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

export const PublicOverview: React.FC = () => {
  const frame = useCurrentFrame();
  const scene = scenes.find((s) => frame >= s.start && frame < s.end) ?? scenes[5];
  const opacity = fade(frame, scene.start, scene.end);
  const local = frame - scene.start;
  const step = scenes.indexOf(scene);
  return (
    <AbsoluteFill style={{background: cream, fontFamily: 'Inter, Arial, sans-serif', color: ink}}>
      <Audio src={staticFile('narration.mp3')} volume={0.95} />
      <div style={{position: 'absolute', top: 48, left: 90, right: 90, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
        <div style={{fontSize: 32, fontWeight: 750, letterSpacing: -1}}>CrossDiffusion</div>
        <div style={{fontSize: 20, letterSpacing: 2, color: green, textTransform: 'uppercase'}}>Crossword agent · one-minute overview</div>
      </div>
      <div style={{position: 'absolute', top: 118, left: 90, right: 90, height: 2, background: '#d8dfd5'}} />
      <div style={{position: 'absolute', top: 185, left: 105, width: 710, opacity, transform: `translateY(${(1 - opacity) * 18}px)`}}>
        <div style={{fontSize: 22, fontWeight: 750, color: green, letterSpacing: 3}}>STEP {scene.number} / 06</div>
        <h1 style={{fontSize: 82, lineHeight: 1.04, letterSpacing: -3.8, margin: '26px 0 28px'}}>{scene.title}</h1>
        <div style={{fontSize: 35, lineHeight: 1.3, color: '#52665b'}}>{scene.detail}</div>
        <div style={{display: 'inline-block', marginTop: 70, background: ink, color: 'white', borderRadius: 18, padding: '22px 34px', fontSize: 24, fontWeight: 700}}>{scene.action} <span style={{marginLeft: 42}}>→</span></div>
      </div>
      <div style={{position: 'absolute', top: 195, left: 950, width: 810, height: 660, borderRadius: 26, background: '#fff', boxShadow: '0 22px 60px #2b4d3822', padding: 38}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 30}}>
          <div style={{fontWeight: 750, fontSize: 25}}>Puzzle view</div>
          <div style={{display: 'flex', gap: 10}}>{scenes.slice(0, 5).map((s, i) => <div key={s.number} style={{width: 20, height: 20, borderRadius: 20, background: i <= step ? green : '#d9e3d8'}} />)}</div>
        </div>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(9, 49px)', gap: 4, position: 'absolute', left: 55, top: 122}}>
          {Array.from({length: 81}, (_, i) => {
            const active = !blocked.has(i);
            const revealed = step >= 3 && active && (i * 7) % 65 < Math.max(0, local / 5);
            const review = step >= 4 && active && i % 17 === 0;
            const scan = step === 1 && Math.floor(local / 7) % 9 === Math.floor(i / 9);
            return <div key={i} style={{width: 49, height: 49, borderRadius: 5, background: !active ? ink : review ? amber : revealed ? green : scan ? mint : '#f3f6ef', border: active ? '2px solid #ccd8cc' : '2px solid '+ink, boxSizing: 'border-box', transform: `scale(${revealed ? 0.97 : 1})`}} />;
          })}
        </div>
        <div style={{position: 'absolute', right: 42, top: 135, width: 150}}>
          {Array.from({length: 7}, (_, i) => <div key={i} style={{height: 13, borderRadius: 9, background: step >= 2 && i === 3 ? amber : '#d6e4d6', marginBottom: 48, width: 95 + (i * 17) % 50, opacity: step >= 1 ? 1 : 0.35}} />)}
        </div>
        <div style={{position: 'absolute', bottom: 35, left: 55, right: 55, display: 'flex', justifyContent: 'space-between', fontSize: 19, color: '#617267'}}>
          <span>{step < 3 ? 'Editable grid and clues' : step === 3 ? 'Iterative first draft' : 'Review changes'}</span>
          <span>{step < 3 ? 'No answer key required' : step === 3 ? 'Confidence-driven' : 'Saved in chat'}</span>
        </div>
      </div>
      <div style={{position: 'absolute', left: 90, right: 90, bottom: 82, display: 'flex', gap: 12}}>
        {scenes.map((s, i) => <div key={s.number} style={{height: 10, flex: 1, borderRadius: 8, background: i < step ? green : i === step ? `linear-gradient(to right, ${green} ${Math.round(local / (s.end - s.start) * 100)}%, #d9e3d8 0)` : '#d9e3d8'}} />)}
      </div>
      <div style={{position: 'absolute', left: 90, bottom: 38, fontSize: 17, color: '#697c70'}}>No puzzle examples or user data appear in this video.</div>
    </AbsoluteFill>
  );
};
