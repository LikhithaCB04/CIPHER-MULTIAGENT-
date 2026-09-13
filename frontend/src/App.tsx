import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './Landing';
import IDE from './IDE';
import CanvasApp from './CanvasApp';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/ide" element={<IDE />} />
        <Route path="/canvas" element={<CanvasApp />} />
      </Routes>
    </Router>
  );
}

export default App;
