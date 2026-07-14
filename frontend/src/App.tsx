import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './Landing';
import IDE from './IDE';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/ide" element={<IDE />} />
      </Routes>
    </Router>
  );
}

export default App;
