import { Workspace } from '@/components/workspace';

/**
 * The application is a single full-viewport workspace. There is no routing:
 * every surface is a panel composited over the canvas, which is what keeps the
 * scene alive across every interaction instead of remounting on navigation.
 */
export default function Page() {
  return <Workspace />;
}
